"""
Small end-to-end DLM example.

This script is intentionally lightweight:
- builds a tiny tokenizer
- creates a small DynLLM
- runs one forward pass
- generates a short continuation
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynllm.tokenizer import DynTokenizer
from dynllm.model import DynLLM, DynLLMConfig


def main() -> None:
    text = (
        "language is a trajectory through state space. "
        "memory bends the path of future thought. "
    )

    tokenizer = DynTokenizer().fit([text])
    cfg = DynLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_state=64,
        use_memory=True,
        use_timescale_sep=True,
        context_window=8,
    )
    model = DynLLM(cfg)

    ids = torch.tensor([tokenizer.encode("language", add_bos=True, add_eos=False)])
    logits = model(ids)
    generated = model.generate(tokenizer.encode("language", add_bos=True, add_eos=False), max_tokens=24)

    print("logits_shape:", tuple(logits.shape))
    print("generated_ids:", generated)
    print("generated_text:", tokenizer.decode(generated))


if __name__ == "__main__":
    main()
