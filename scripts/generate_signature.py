from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "SIGNATURE.sha256"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
SKIP_FILES = {"SIGNATURE.sha256", ".DS_Store"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if name in SKIP_FILES:
                continue
            yield Path(dirpath) / name


def main() -> None:
    lines = []
    for path in iter_files(ROOT):
        rel = path.relative_to(ROOT)
        lines.append(f"{sha256_file(path)}  {rel.as_posix()}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
