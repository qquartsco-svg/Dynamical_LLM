from __future__ import annotations

from pathlib import Path

from dynllm import __version__


ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_version_file():
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == __version__


def test_required_files_exist():
    required = [
        ROOT / "README.md",
        ROOT / "README_EN.md",
        ROOT / "pyproject.toml",
        ROOT / "train.py",
        ROOT / "generate.py",
    ]
    for path in required:
        assert path.exists(), f"missing {path.name}"
