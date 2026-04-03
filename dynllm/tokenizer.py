"""
L0: Token Interface — 토크나이저

두 가지 모드:
  1. DynTokenizer (char-level) — 코퍼스에서 문자 vocab 학습. 간단, 투명.
  2. ByteTokenizer (byte-level) — 고정 256 vocab. fit 불필요, 모든 언어 즉시 대응.

ByteTokenizer는 한국어·중국어·일본어 등 멀티바이트 문자를
UTF-8 바이트로 분해해서 토큰 수를 줄인다.
  예: char-level "안녕" = 2 tokens, byte-level "안녕" = 6 bytes
  하지만 char-level에서 vocab이 수천으로 커지는 문제를 고정 256으로 해결.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_OFFSET = 4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Character-level Tokenizer (original)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DynTokenizer:
    """Character-level tokenizer with special tokens."""

    char_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_char: Dict[int, str] = field(default_factory=dict)
    vocab_size: int = SPECIAL_OFFSET

    @property
    def mode(self) -> str:
        return "char"

    def fit(self, texts: List[str]) -> "DynTokenizer":
        chars = set()
        for text in texts:
            chars.update(text)
        for i, ch in enumerate(sorted(chars)):
            idx = i + SPECIAL_OFFSET
            self.char_to_id[ch] = idx
            self.id_to_char[idx] = ch
        self.vocab_size = len(self.char_to_id) + SPECIAL_OFFSET
        self.id_to_char[PAD_ID] = "<PAD>"
        self.id_to_char[BOS_ID] = "<BOS>"
        self.id_to_char[EOS_ID] = "<EOS>"
        self.id_to_char[UNK_ID] = "<UNK>"
        return self

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(BOS_ID)
        for ch in text:
            ids.append(self.char_to_id.get(ch, UNK_ID))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: List[int]) -> str:
        chars = []
        for idx in ids:
            if idx in (PAD_ID, BOS_ID, EOS_ID):
                continue
            chars.append(self.id_to_char.get(idx, "?"))
        return "".join(chars)

    def save(self, path: Path) -> None:
        data = {
            "type": "char",
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DynTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") == "byte":
            return ByteTokenizer.load(path)
        tok = cls()
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(v): k for k, v in data["char_to_id"].items()}
        tok.id_to_char[PAD_ID] = "<PAD>"
        tok.id_to_char[BOS_ID] = "<BOS>"
        tok.id_to_char[EOS_ID] = "<EOS>"
        tok.id_to_char[UNK_ID] = "<UNK>"
        tok.vocab_size = data["vocab_size"]
        return tok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Byte-level Tokenizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BYTE_VOCAB_SIZE = 256 + SPECIAL_OFFSET   # 0-255 byte values + 4 special tokens


@dataclass
class ByteTokenizer:
    """
    Byte-level tokenizer — fixed 260 vocab (256 bytes + 4 special).

    fit()이 필요 없다. 모든 UTF-8 텍스트를 바이트 시퀀스로 직접 처리.
    한국어 한 글자 = 3 UTF-8 바이트이지만, vocab이 260으로 고정되므로
    char-level에서 한국어 vocab이 수천으로 폭발하는 문제가 없다.
    """

    vocab_size: int = BYTE_VOCAB_SIZE

    @property
    def mode(self) -> str:
        return "byte"

    def fit(self, texts: List[str]) -> "ByteTokenizer":
        return self

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for b in text.encode("utf-8"):
            ids.append(b + SPECIAL_OFFSET)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: List[int]) -> str:
        raw: List[int] = []
        for idx in ids:
            if idx < SPECIAL_OFFSET:
                continue
            byte_val = idx - SPECIAL_OFFSET
            if 0 <= byte_val <= 255:
                raw.append(byte_val)
        return bytes(raw).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        data = {"type": "byte", "vocab_size": self.vocab_size}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.vocab_size = data.get("vocab_size", BYTE_VOCAB_SIZE)
        return tok


def load_tokenizer(path: Path) -> "DynTokenizer | ByteTokenizer":
    """Unified loader — detects type from saved JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "byte":
        return ByteTokenizer.load(path)
    return DynTokenizer.load(path)
