"""
MemoryRank 어댑터 — PageRank 기반 기억 중요도 + 재정렬

DynLLM의 메모리 시스템에 그래프 기반 중요도 랭킹을 추가한다.

핵심 원리:
  기존: cosine(cue, patterns) → top-k 선택 (사전식 검색)
  개선: cosine 후보 → PageRank 중요도 재정렬 → 최종 top-k (구글링)

MemoryGraph는 저장된 기억 패턴 간 관계를 추적:
  - 시간적 인접: 연속 저장된 패턴 간 에지
  - 유사도: 새 패턴과 기존 패턴 cosine > threshold → 에지
  - 재활성: recall된 패턴의 frequency 가중치 상승

Cognitive_Kernel의 MemoryRankEngine(v1.1.0)과 동일한 Personalized PageRank
알고리즘을 torch 기반으로 자체 구현한다. (독립 실행 원칙)

수식:
  r_{t+1} = α · M · r_t + (1 − α) · v
  α = damping (0.85)
  M = column-stochastic 전이 행렬
  v = personalization 벡터 (recency, frequency, importance 가중)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MemoryNodeAttrs:
    """기억 노드 속성 (PageRank personalization에 반영)."""
    recency: float = 0.0
    frequency: float = 0.0
    importance: float = 0.5

    def score(self, w_recency: float = 1.0, w_freq: float = 1.0) -> float:
        total = w_recency * self.recency + w_freq * self.frequency + self.importance
        return max(total, 1e-8)


@dataclass
class MemoryGraphConfig:
    damping: float = 0.85
    max_iter: int = 50
    tol: float = 1e-6
    similarity_threshold: float = 0.3
    recency_decay: float = 0.95
    rerank_alpha: float = 0.5  # cosine vs pagerank blend


class MemoryGraph:
    """
    기억 패턴 간 관계 그래프 + Personalized PageRank 랭킹.

    저장 시 → 에지 자동 생성 (시간적 인접 + 유사도)
    검색 시 → cosine 후보를 PageRank로 재정렬
    """

    def __init__(self, config: Optional[MemoryGraphConfig] = None):
        self.config = config or MemoryGraphConfig()
        self._nodes: Dict[str, MemoryNodeAttrs] = {}
        self._edges: List[Tuple[str, str, float]] = []
        self._ranks: Optional[Dict[str, float]] = None
        self._dirty = True
        self._last_node_id: Optional[str] = None

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    def add_node(self, node_id: str, attrs: Optional[MemoryNodeAttrs] = None):
        self._nodes[node_id] = attrs or MemoryNodeAttrs()
        self._dirty = True

    def add_edge(self, src: str, dst: str, weight: float = 1.0):
        if src in self._nodes and dst in self._nodes and src != dst:
            self._edges.append((src, dst, weight))
            self._dirty = True

    def register_pattern(
        self,
        node_id: str,
        pattern: torch.Tensor,
        existing_patterns: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        새 패턴 등록 — 노드 생성 + 시간적/유사도 에지 자동 생성.

        Args:
            node_id: 노드 식별자 (e.g. "heb_42")
            pattern: 패턴 벡터
            existing_patterns: {node_id: tensor} 기존 저장 패턴
        """
        for nid in self._nodes:
            self._nodes[nid].recency *= self.config.recency_decay

        self.add_node(node_id, MemoryNodeAttrs(recency=1.0, frequency=0.0, importance=0.5))

        if self._last_node_id is not None and self._last_node_id in self._nodes:
            self.add_edge(self._last_node_id, node_id, weight=1.0)

        if existing_patterns is not None and len(existing_patterns) > 0:
            p_norm = F.normalize(pattern.detach().float().flatten().unsqueeze(0), dim=-1)
            for eid, ep in existing_patterns.items():
                if eid == node_id:
                    continue
                e_norm = F.normalize(ep.detach().float().flatten().unsqueeze(0), dim=-1)
                sim = (p_norm * e_norm).sum().item()
                if sim > self.config.similarity_threshold:
                    self.add_edge(node_id, eid, weight=sim)
                    self.add_edge(eid, node_id, weight=sim)

        self._last_node_id = node_id

    def bump_frequency(self, node_id: str):
        if node_id in self._nodes:
            self._nodes[node_id].frequency = min(1.0, self._nodes[node_id].frequency + 0.1)
            self._dirty = True

    def compute_pagerank(self) -> Dict[str, float]:
        """
        Personalized PageRank.

        r_{t+1} = α · M · r_t + (1 − α) · v
        """
        if not self._dirty and self._ranks is not None:
            return self._ranks

        n = len(self._nodes)
        if n == 0:
            self._ranks = {}
            self._dirty = False
            return self._ranks

        id_to_idx = {nid: i for i, nid in enumerate(self._nodes)}
        idx_to_id = {i: nid for nid, i in id_to_idx.items()}

        M = torch.zeros(n, n)
        for src, dst, w in self._edges:
            if src in id_to_idx and dst in id_to_idx:
                M[id_to_idx[dst], id_to_idx[src]] += w

        col_sums = M.sum(dim=0)
        dangling = col_sums < 1e-12
        col_sums = col_sums.clamp(min=1e-12)
        M = M / col_sums.unsqueeze(0)
        M[:, dangling] = 1.0 / n

        v = torch.zeros(n)
        for nid, attrs in self._nodes.items():
            v[id_to_idx[nid]] = attrs.score()
        v_sum = v.sum()
        v = v / v_sum if v_sum > 0 else torch.ones(n) / n

        alpha = self.config.damping
        r = torch.ones(n) / n

        for _ in range(self.config.max_iter):
            r_new = alpha * (M @ r) + (1.0 - alpha) * v
            if (r_new - r).abs().sum().item() < self.config.tol:
                r = r_new
                break
            r = r_new

        r_sum = r.sum()
        if r_sum > 0:
            r = r / r_sum

        self._ranks = {idx_to_id[i]: r[i].item() for i in range(n)}
        self._dirty = False
        return self._ranks

    def get_top_k(self, k: int) -> List[Tuple[str, float]]:
        ranks = self.compute_pagerank()
        return sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:k]

    def rerank_candidates(
        self,
        candidate_ids: List[str],
        similarity_scores: List[float],
    ) -> List[Tuple[str, float]]:
        """
        cosine 후보를 PageRank로 재정렬.

        최종 점수 = alpha * similarity + (1 - alpha) * pagerank
        """
        ranks = self.compute_pagerank()
        alpha = self.config.rerank_alpha

        combined = []
        for cid, sim in zip(candidate_ids, similarity_scores):
            pr = ranks.get(cid, 0.0)
            score = alpha * sim + (1.0 - alpha) * pr
            combined.append((cid, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    def diagnostics(self) -> dict:
        ranks = self.compute_pagerank()
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "top_5": self.get_top_k(5),
            "rank_entropy": _entropy_from_dict(ranks) if ranks else 0.0,
        }

    def state_dict(self) -> dict:
        return {
            "nodes": {
                k: {"recency": v.recency, "frequency": v.frequency, "importance": v.importance}
                for k, v in self._nodes.items()
            },
            "edges": self._edges,
            "last_node_id": self._last_node_id,
        }

    def load_state_dict(self, state: dict):
        self._nodes = {
            k: MemoryNodeAttrs(**v)
            for k, v in state.get("nodes", {}).items()
        }
        self._edges = [(s, d, w) for s, d, w in state.get("edges", [])]
        self._last_node_id = state.get("last_node_id")
        self._dirty = True


def _entropy_from_dict(d: Dict[str, float]) -> float:
    """값 분포의 Shannon 엔트로피."""
    import math
    vals = list(d.values())
    total = sum(vals)
    if total <= 0:
        return 0.0
    h = 0.0
    for v in vals:
        p = v / total
        if p > 1e-12:
            h -= p * math.log(p)
    return h
