"""
L0: Token Interface — 문자 수준 토크나이저

토큰을 단순 id가 아니라 input impulse로 취급한다.
Phase A에서는 char-level로 시작. 이후 subword/byte 확장 가능.
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


@dataclass
class DynTokenizer:
    """Character-level tokenizer with special tokens."""

    char_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_char: Dict[int, str] = field(default_factory=dict)
    vocab_size: int = SPECIAL_OFFSET

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
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DynTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(v): k for k, v in data["char_to_id"].items()}
        tok.id_to_char[PAD_ID] = "<PAD>"
        tok.id_to_char[BOS_ID] = "<BOS>"
        tok.id_to_char[EOS_ID] = "<EOS>"
        tok.id_to_char[UNK_ID] = "<UNK>"
        tok.vocab_size = data["vocab_size"]
        return tok
