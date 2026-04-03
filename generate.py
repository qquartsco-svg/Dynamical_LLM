"""
Dynamical LLM Foundation — 추론/생성 엔트리포인트

저장된 모델을 로드해서 텍스트를 생성한다.
모델이 없으면 즉석으로 작은 모델을 부팅하여 생성한다.

사용법:
  python generate.py "hello"                # 프롬프트로 생성
  python generate.py --model path/model.pt  # 저장된 모델 사용
  python generate.py --interactive          # 인터랙티브 모드
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynllm.tokenizer import DynTokenizer
from dynllm.model import DynLLM, DynLLMConfig


def load_or_create(
    model_path: str = None,
    tokenizer_path: str = None,
) -> tuple[DynLLM, DynTokenizer]:
    """모델+토크나이저 로드. 없으면 기본 생성."""
    if model_path and Path(model_path).exists():
        model = DynLLM.load(Path(model_path))
        tok_p = Path(tokenizer_path) if tokenizer_path else Path(model_path).parent / "tokenizer.json"
        if tok_p.exists():
            tok = DynTokenizer.load(tok_p)
        else:
            tok = DynTokenizer().fit(["abcdefghijklmnopqrstuvwxyz .,'!?\n"])
        return model, tok

    print("[No saved model — creating minimal untrained model]")
    tok = DynTokenizer().fit(["abcdefghijklmnopqrstuvwxyz .,'!?\n"])
    cfg = DynLLMConfig(vocab_size=tok.vocab_size, d_state=64, use_memory=False)
    model = DynLLM(cfg)
    return model, tok


def generate_text(
    model: DynLLM,
    tokenizer: DynTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 0,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    gen_ids = model.generate(
        prompt_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(gen_ids)


def interactive(model: DynLLM, tokenizer: DynTokenizer):
    print()
    print("━━━ Dynamical LLM — Interactive ━━━")
    print("Type a prompt, or 'exit' to quit.")
    print()

    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt or prompt.lower() in ("exit", "quit"):
            break

        result = generate_text(model, tokenizer, prompt)
        print(f"DynLLM > {result}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Dynamical LLM — Generate")
    parser.add_argument("prompt", nargs="*", help="Prompt text")
    parser.add_argument("--model", type=str, default=None, help="Path to model.pt")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model, tok = load_or_create(args.model, args.tokenizer)

    if args.interactive:
        interactive(model, tok)
        return

    if args.prompt:
        prompt = " ".join(args.prompt)
        result = generate_text(
            model, tok, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"{result}")
    else:
        interactive(model, tok)


if __name__ == "__main__":
    main()
