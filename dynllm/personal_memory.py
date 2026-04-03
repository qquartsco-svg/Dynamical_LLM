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
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                src_crystal_id INTEGER NOT NULL,
                dst_crystal_id INTEGER NOT NULL,
                weight REAL DEFAULT 1.0,
                link_type TEXT DEFAULT 'temporal',
                FOREIGN KEY (src_crystal_id) REFERENCES crystals(id),
                FOREIGN KEY (dst_crystal_id) REFERENCES crystals(id)
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

    def link_crystals(self, src_id: int, dst_id: int, weight: float = 1.0, link_type: str = "temporal"):
        """두 crystal 간 관계 에지 생성."""
        self._conn.execute(
            "INSERT INTO memory_links (src_crystal_id, dst_crystal_id, weight, link_type) VALUES (?, ?, ?, ?)",
            (src_id, dst_id, weight, link_type),
        )
        self._conn.commit()

    def recall_crystals_ranked(self, top_k: int = 10) -> list[dict]:
        """
        PageRank 기반 crystal 재정렬 조회.

        memory_links 테이블의 에지로 그래프를 구성하고
        MemoryGraph(PageRank)로 중요도를 재계산한다.
        """
        from .memory_rank_adapter import MemoryGraph, MemoryNodeAttrs

        all_crystals = self._conn.execute(
            "SELECT id, pattern, importance, access_count FROM crystals"
        ).fetchall()
        if not all_crystals:
            return []

        graph = MemoryGraph()
        now = time.time()

        for c in all_crystals:
            cid, pattern, importance, access_count = c
            created_row = self._conn.execute(
                "SELECT created_at FROM crystals WHERE id = ?", (cid,)
            ).fetchone()
            created_at = created_row[0] if created_row and created_row[0] else now

            age_hours = max((now - created_at) / 3600.0, 0.01)
            recency = 1.0 / (1.0 + age_hours / 24.0)
            freq = min(1.0, (access_count or 0) / 10.0)

            graph.add_node(
                f"crystal_{cid}",
                MemoryNodeAttrs(recency=recency, frequency=freq, importance=importance),
            )

        links = self._conn.execute(
            "SELECT src_crystal_id, dst_crystal_id, weight FROM memory_links"
        ).fetchall()
        for src, dst, w in links:
            graph.add_edge(f"crystal_{src}", f"crystal_{dst}", w)

        ranks = graph.compute_pagerank()

        crystal_map = {c[0]: c for c in all_crystals}
        scored = []
        for cid, row in crystal_map.items():
            pr = ranks.get(f"crystal_{cid}", 0.0)
            scored.append((cid, row[1], row[2], row[3], pr))

        scored.sort(key=lambda x: x[4], reverse=True)

        return [
            {"id": s[0], "pattern": s[1], "importance": s[2], "access_count": s[3], "pagerank": s[4]}
            for s in scored[:top_k]
        ]

    def bump_access_count(self, crystal_id: int):
        """crystal 접근 횟수 증가."""
        self._conn.execute(
            "UPDATE crystals SET access_count = access_count + 1 WHERE id = ?",
            (crystal_id,),
        )
        self._conn.commit()

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
