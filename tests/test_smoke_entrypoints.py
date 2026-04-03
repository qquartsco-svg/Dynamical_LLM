"""Smoke tests for train.py and generate.py entrypoints."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from train import train
from generate import load_or_create, generate_text


def test_generate_bootstrap_smoke():
    model, tok = load_or_create()
    out = generate_text(model, tok, "hi", max_tokens=8, temperature=0.7)
    assert isinstance(out, str)


def test_train_smoke(tmp_path: Path):
    text = "abc abc abc abc abc " * 6
    model, tok = train(
        text=text,
        epochs=1,
        d_state=16,
        seq_len=8,
        batch_size=2,
        lr=1e-3,
        save_dir=tmp_path,
        use_timescale=False,
        context_window=4,
    )
    assert model.count_parameters() > 0
    assert tok.vocab_size >= 4
    assert (tmp_path / "final_model.pt").exists()
    assert (tmp_path / "tokenizer.json").exists()
