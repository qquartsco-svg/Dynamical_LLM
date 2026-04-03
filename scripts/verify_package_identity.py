from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REQUIRED = [
    ROOT / "pyproject.toml",
    ROOT / "VERSION",
    ROOT / "README.md",
    ROOT / "README_EN.md",
    ROOT / "dynllm" / "__init__.py",
    ROOT / "dynllm" / "model.py",
    ROOT / "dynllm" / "dynamics_core.py",
    ROOT / "train.py",
    ROOT / "generate.py",
    ROOT / "tests" / "test_dynllm.py",
]


def main() -> int:
    missing = [p.relative_to(ROOT).as_posix() for p in REQUIRED if not p.exists()]
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
