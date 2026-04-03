"""
Dynamical LLM Foundation — 학습 스크립트

Phase B: 실제 텍스트 코퍼스 + 체크포인트 + Phase B dynamics 검증.

사용법:
  python train.py                           # 기본 학습 (내장 텍스트)
  python train.py --corpus data/input.txt   # 파일 코퍼스
  python train.py --epochs 100 --save out/  # 체크포인트 저장
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynllm.tokenizer import DynTokenizer
from dynllm.model import DynLLM, DynLLMConfig


class CharDataset(Dataset):
    def __init__(self, text: str, tokenizer: DynTokenizer, seq_len: int = 64):
        self.ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.ids[idx: idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_corpus(path: str | None) -> str:
    """파일이면 읽고, 없으면 기본 텍스트 반환."""
    if path and Path(path).exists():
        text = Path(path).read_text(encoding="utf-8")
        print(f"Corpus: {path} ({len(text):,} chars)")
        return text

    default = (
        "the brain is a dynamical system that evolves over time. "
        "each thought is a trajectory through state space. "
        "memory is not storage but pattern completion. "
        "learning is not weight update but adaptation of dynamics. "
        "the mind is not a computer but a river. "
        "consciousness emerges from the coupling of fast and slow processes. "
        "attention is not a spotlight but a dynamical gate. "
        "language flows through continuous state evolution. "
        "the past pulls the present through associative memory. "
        "stability and plasticity must coexist in every mind. "
    ) * 30
    print(f"Corpus: built-in ({len(default):,} chars)")
    return default


def train(
    text: str,
    epochs: int = 50,
    d_state: int = 128,
    seq_len: int = 64,
    batch_size: int = 16,
    lr: float = 3e-4,
    save_dir: Path = None,
    use_timescale: bool = True,
    context_window: int = 16,
):
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Device: {device_name}")

    tokenizer = DynTokenizer().fit([text])
    print(f"Vocab: {tokenizer.vocab_size} chars")

    dataset = CharDataset(text, tokenizer, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} samples, seq_len={seq_len}")

    cfg = DynLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_state=d_state,
        use_memory=True,
        use_adaptation=False,
        context_window=context_window,
        use_timescale_sep=use_timescale,
    )
    model = DynLLM(cfg).to(device)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters")
    print(f"  Dynamics: d_state={d_state}, substeps={cfg.n_substeps}, damping={cfg.damping}")
    print(f"  Phase B: context_window={context_window}, timescale_sep={use_timescale}")
    if use_timescale:
        print(f"  Timescale: fast_ratio={cfg.fast_ratio}, τ_fast={cfg.tau_fast}, τ_slow={cfg.tau_slow}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float("inf")
    t0 = time.time()
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        loss_history.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            damping_val = model.dynamics.damping.item()

            tau_info = ""
            if use_timescale and hasattr(model.dynamics, '_log_tau_fast'):
                tau_f = torch.exp(model.dynamics._log_tau_fast).item()
                tau_s = torch.exp(model.dynamics._log_tau_slow).item()
                tau_info = f"τf={tau_f:.3f} τs={tau_s:.3f} | "

            print(
                f"  epoch {epoch:3d}/{epochs} | "
                f"loss={avg_loss:.4f} | ppl={ppl:.1f} | "
                f"λ={damping_val:.4f} | {tau_info}"
                f"grad={grad_norm:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f} | "
                f"{elapsed:.1f}s"
            )

            model.eval()
            prompt = text[:5]
            prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            gen_ids = model.generate(prompt_ids, max_tokens=40, temperature=0.7)
            gen_text = tokenizer.decode(gen_ids)
            print(f"         sample: \"{prompt}\" → \"{gen_text}\"")
            print()

        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_dir:
                model.save(save_dir / "best_model.pt")
                tokenizer.save(save_dir / "tokenizer.json")

        if save_dir and epoch % 10 == 0:
            ckpt_path = save_dir / f"checkpoint_epoch{epoch}.pt"
            model.save(ckpt_path)

    elapsed = time.time() - t0
    print(f"━━━ Training complete ━━━")
    print(f"  Best loss: {best_loss:.4f} (ppl={math.exp(min(best_loss,20)):.1f})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Parameters: {n_params:,}")

    if loss_history:
        improvement = loss_history[0] - loss_history[-1]
        print(f"  Loss drop: {improvement:+.4f} (first={loss_history[0]:.4f} → last={loss_history[-1]:.4f})")

    if save_dir:
        model.save(save_dir / "final_model.pt")
        tokenizer.save(save_dir / "tokenizer.json")
        print(f"  Saved: {save_dir}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Dynamical LLM Foundation — Train")
    parser.add_argument("--corpus", type=str, default=None, help="Path to text corpus file")
    parser.add_argument("--text", type=str, default=None, help="Inline training text")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--context-window", type=int, default=16)
    parser.add_argument("--no-timescale", action="store_true", help="Disable fast/slow separation")
    args = parser.parse_args()

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    text = args.text or load_corpus(args.corpus)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Dynamical LLM Foundation — Phase B Training")
    print("  ẋ = −λx + F(x, u, m, c; θ)")
    print("  + Context Coupling + Timescale Separation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    train(
        text=text,
        epochs=args.epochs,
        d_state=args.d_state,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=save_dir,
        use_timescale=not args.no_timescale,
        context_window=args.context_window,
    )


if __name__ == "__main__":
    main()
