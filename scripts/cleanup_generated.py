from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    removed = 0
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
            removed += 1
    for name in [".pytest_cache", ".DS_Store"]:
        for path in ROOT.rglob(name):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            removed += 1
    print(f"removed={removed}")


if __name__ == "__main__":
    main()
