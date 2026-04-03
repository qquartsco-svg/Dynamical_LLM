from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SIGNATURE = ROOT / "SIGNATURE.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not SIGNATURE.exists():
        print("SIGNATURE.sha256 not found")
        return 1

    passed = 0
    failed = 0
    missing = 0
    for line in SIGNATURE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.exists():
            print(f"MISSING  {rel}")
            missing += 1
            continue
        if sha256_file(path) != digest:
            print(f"FAIL     {rel}")
            failed += 1
        else:
            passed += 1
    print(f"passed={passed} failed={failed} missing={missing}")
    return 0 if failed == 0 and missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
