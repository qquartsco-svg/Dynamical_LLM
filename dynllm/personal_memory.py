"""
개인 메모리 누적 시스템 (PersonalMemoryStore)

사용자와의 상호작용을 영구 저장하고, DynLLM의 메모리에 주입한다.

3단계 구조:
  1. Interaction Log: 모든 대화 기록 (SQLite)
  2. Knowledge Crystals: 중요 패턴 추출·압축 (벡터)
  3. Memory Injection: DynLLM의 Hebbian/Episodic에 주입

Atom 에이전트가 대화할 때마다 여기에 축적되고,
DynLLM이 재학습 없이도 개인화된 응답을 생성할 수 있다.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class Interaction:
    """단일 상호작용 기록."""
    user_input: str
    response: str
    timestamp: float = field(default_factory=time.time)
    source: str = "user"
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)


class PersonalMemoryStore:
    """
    SQLite 기반 영구 개인 메모리.

    대화 이력을 저장하고, 중요 패턴을 벡터로 추출하여
    DynLLM의 메모리 시스템에 주입할 수 있다.
    """

    def __init__(self, db_path: Path | str = "~/.dynllm/personal_memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp REAL,
                source TEXT DEFAULT 'user',
                confidence REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]'
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS crystals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                vector BLOB,
                importance REAL DEFAULT 0.5,
                created_at REAL,
                access_count INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()

    def log(self, interaction: Interaction) -> int:
        cur = self._conn.execute(
            "INSERT INTO interactions (user_input, response, timestamp, source, confidence, tags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (interaction.user_input, interaction.response, interaction.timestamp,
             interaction.source, interaction.confidence, json.dumps(interaction.tags)),
        )
        self._conn.commit()
        return cur.lastrowid

    def store_crystal(self, pattern: str, vector: torch.Tensor, importance: float = 0.5) -> int:
        """중요 패턴을 벡터와 함께 저장."""
        vec_bytes = vector.detach().cpu().numpy().tobytes()
        cur = self._conn.execute(
            "INSERT INTO crystals (pattern, vector, importance, created_at) VALUES (?, ?, ?, ?)",
            (pattern, vec_bytes, importance, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def recall_crystals(self, top_k: int = 10) -> list[dict]:
        """가장 중요한 knowledge crystal 조회."""
        rows = self._conn.execute(
            "SELECT id, pattern, importance, access_count FROM crystals "
            "ORDER BY importance DESC, access_count DESC LIMIT ?",
            (top_k,),
        ).fetchall()
        return [
            {"id": r[0], "pattern": r[1], "importance": r[2], "access_count": r[3]}
            for r in rows
        ]

    def recent_interactions(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT user_input, response, timestamp, source, confidence "
            "FROM interactions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"user_input": r[0], "response": r[1], "timestamp": r[2],
             "source": r[3], "confidence": r[4]}
            for r in rows
        ]

    def stats(self) -> dict:
        n_interactions = self._conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        n_crystals = self._conn.execute("SELECT COUNT(*) FROM crystals").fetchone()[0]
        return {
            "n_interactions": n_interactions,
            "n_crystals": n_crystals,
            "db_path": str(self.db_path),
        }

    def close(self):
        self._conn.close()


class MemoryInjector:
    """
    PersonalMemoryStore → DynLLM 메모리 주입기.

    저장된 knowledge crystal의 벡터를 DynLLM의 Hebbian memory에 주입하거나,
    최근 대화 이력을 episodic memory에 에피소드로 변환·저장한다.
    """

    def __init__(self, store: PersonalMemoryStore):
        self.store = store

    def inject_crystals_to_hebbian(self, hebbian_memory, d_state: int, top_k: int = 10):
        """knowledge crystals를 Hebbian memory에 주입."""
        import numpy as np

        crystals = self.store.recall_crystals(top_k=top_k)
        injected = 0
        for c in crystals:
            row = self.store._conn.execute(
                "SELECT vector FROM crystals WHERE id = ?", (c["id"],)
            ).fetchone()
            if row and row[0]:
                vec = torch.from_numpy(
                    np.frombuffer(row[0], dtype=np.float32)
                )
                if vec.shape[0] == d_state:
                    hebbian_memory.store(vec)
                    injected += 1
        return injected

    def inject_recent_to_episodic(self, episodic_memory, encoder, tokenizer, n_recent: int = 5):
        """최근 대화를 episodic memory에 에피소드로 주입."""
        interactions = self.store.recent_interactions(limit=n_recent)
        injected = 0
        for ix in interactions:
            text = ix["user_input"] + " " + ix["response"]
            ids = tokenizer.encode(text[:200], add_bos=False, add_eos=False)
            if len(ids) < 3:
                continue
            id_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                impulses = encoder(id_tensor).squeeze(0)
            episodic_memory.store_episode(impulses)
            injected += 1
        return injected
