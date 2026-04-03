"""
L3: Memory — 4단 기억 시스템

Phase C 고도화:
  1. Short-term: 은닉 상태 x 자체 (−λx로 자연 감쇠) → dynamics_core
  2. Working memory: PFC형 슬롯 + write gate + dynamics feedback
  3. Long-term (Hebbian): selective recall + 개별 저장
  4. Episodic: 시퀀스 단위 에피소드 기억 → 맥락 복원

Phase G: MemoryRank 통합
  - MemoryGraph 기반 PageRank 재정렬
  - 저장 시 에지 자동 생성, 검색 시 중요도 재정렬

Working Memory ↔ Dynamics Core 양방향:
  - read: dynamics가 working memory를 조회
  - write: dynamics가 working memory에 기록
  - feedback: working memory 요약이 dynamics 입력에 다시 주입
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryConfig:
    d_state: int = 128
    n_working_slots: int = 8
    working_decay: float = 0.95
    write_threshold: float = 0.3
    hebbian_size: int = 256
    hebbian_lr: float = 0.01

    # Phase C
    n_episodes: int = 16            # episodic memory 에피소드 수
    episode_len: int = 32           # 에피소드당 상태 수
    selective_top_k: int = 3        # selective recall 상위 k개


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Working Memory (Phase C: dynamics feedback 추가)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WorkingMemory(nn.Module):
    """
    PFC형 작업 기억 + Write Gate + Dynamics Feedback.

    Phase C 추가:
    - summary(): 현재 작업기억 요약 벡터를 dynamics core에 피드백
    - importance weighting: 활성화 강도에 비례한 가중 요약
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.n_slots = cfg.n_working_slots
        self.d_state = cfg.d_state
        self.decay = cfg.working_decay
        self.write_threshold = cfg.write_threshold

        self.query_proj = nn.Linear(cfg.d_state, cfg.d_state, bias=False)
        self.key_proj = nn.Linear(cfg.d_state, cfg.d_state, bias=False)
        self.write_gate_proj = nn.Linear(cfg.d_state * 2, 1)

        self.feedback_proj = nn.Linear(cfg.d_state, cfg.d_state, bias=False)

    def init_state(self, batch_size: int, device: torch.device):
        slots = torch.zeros(batch_size, self.n_slots, self.d_state, device=device)
        activations = torch.zeros(batch_size, self.n_slots, device=device)
        return slots, activations

    def read(
        self,
        query: torch.Tensor,
        slots: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(slots)
        scale = math.sqrt(self.d_state)
        attn = (q * k).sum(-1) / scale
        attn = attn * activations
        weights = F.softmax(attn, dim=-1)
        return (weights.unsqueeze(-1) * slots).sum(dim=1)

    def summary(self, slots: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        """
        작업기억 가중 요약 → dynamics core로 피드백.
        활성화가 높은 슬롯에 더 큰 가중치.
        """
        weights = F.softmax(activations, dim=-1)
        weighted = (weights.unsqueeze(-1) * slots).sum(dim=1)
        return self.feedback_proj(weighted)

    def _compute_novelty(
        self, state: torch.Tensor, slots: torch.Tensor, activations: torch.Tensor,
    ) -> torch.Tensor:
        active_mask = (activations > 0.01).float()
        state_norm = F.normalize(state, dim=-1).unsqueeze(1)
        slots_norm = F.normalize(slots, dim=-1)
        similarity = (state_norm * slots_norm).sum(dim=-1) * active_mask
        max_sim = similarity.max(dim=-1).values
        return 1.0 - max_sim.clamp(0, 1)

    def write(
        self, state: torch.Tensor, slots: torch.Tensor, activations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_acts = activations * self.decay
        novelty = self._compute_novelty(state, slots, new_acts)

        wm_summary = (new_acts.unsqueeze(-1) * slots).sum(dim=1) / slots.shape[1]
        gate_input = torch.cat([state, wm_summary], dim=-1)
        gate_score = torch.sigmoid(self.write_gate_proj(gate_input)).squeeze(-1)

        should_write = ((novelty > self.write_threshold) | (gate_score > 0.5)).float()

        weakest = new_acts.argmin(dim=-1)
        batch_idx = torch.arange(state.shape[0], device=state.device)

        new_slots = slots.clone()
        new_acts_out = new_acts.clone()

        write_mask = should_write.bool()
        if write_mask.any():
            new_slots[batch_idx[write_mask], weakest[write_mask]] = state[write_mask].detach()
            new_acts_out[batch_idx[write_mask], weakest[write_mask]] = 1.0

        return new_slots, new_acts_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Hebbian Memory (Phase C: Selective Recall)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HebbianMemory(nn.Module):
    """
    장기 연상 기억 — Hippocampus 헤비안 규칙.

    Phase C 추가:
    - selective_recall(): 저장된 패턴 중 cue와 가장 유사한 top-k만 선택 회수
    - pattern_store: 저장된 개별 패턴 기록 (sparse retrieval용)
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.d_state = cfg.d_state
        self.capacity = cfg.hebbian_size
        self.lr = cfg.hebbian_lr
        self.top_k = cfg.selective_top_k

        self.register_buffer("W", torch.zeros(cfg.d_state, cfg.d_state))
        self.register_buffer("n_stored", torch.tensor(0))

        self.register_buffer(
            "pattern_store",
            torch.zeros(cfg.hebbian_size, cfg.d_state),
        )
        self.register_buffer("store_ptr", torch.tensor(0))

        self.readout_proj = nn.Linear(cfg.d_state, cfg.d_state, bias=False)
        self.selective_gate = nn.Linear(cfg.d_state * 2, cfg.d_state)

    def store(self, pattern: torch.Tensor) -> None:
        with torch.no_grad():
            if pattern.dim() > 1:
                for i in range(pattern.shape[0]):
                    self._store_single(pattern[i])
            else:
                self._store_single(pattern)

    def _store_single(self, p: torch.Tensor) -> None:
        p_norm = F.normalize(p.detach(), dim=0)
        if p_norm.norm().item() < 1e-6:
            return

        update = self.lr * torch.outer(p_norm, p_norm)
        if self.n_stored.item() > self.capacity:
            self.W.mul_(self.capacity / (self.capacity + 1))
        self.W.add_(update)

        ptr = self.store_ptr.item() % self.capacity
        self.pattern_store[ptr] = p_norm
        self.store_ptr.add_(1)
        self.n_stored.add_(1)

    def recall(self, cue: torch.Tensor) -> torch.Tensor:
        """기본 연상 회수 (전체 W 사용)."""
        W_snapshot = self.W.detach().clone()
        raw = F.linear(cue, W_snapshot)
        return self.readout_proj(torch.tanh(raw))

    def selective_recall(self, cue: torch.Tensor) -> torch.Tensor:
        """
        Selective recall: 저장된 개별 패턴 중 cue와 가장 유사한 top-k만 회수.
        sparse retrieval — 전체 W를 쓰지 않고 관련 패턴만 선택.
        """
        n_valid = min(self.n_stored.item(), self.capacity)
        if n_valid == 0:
            return self.readout_proj(torch.zeros_like(cue))

        valid_patterns = self.pattern_store[:n_valid]

        cue_norm = F.normalize(cue, dim=-1)
        pattern_norm = F.normalize(valid_patterns, dim=-1)
        similarity = cue_norm @ pattern_norm.T

        k = min(self.top_k, n_valid)
        top_vals, top_idx = similarity.topk(k, dim=-1)

        weights = F.softmax(top_vals, dim=-1)
        selected = valid_patterns[top_idx.squeeze(0)] if cue.shape[0] == 1 else self._batch_select(valid_patterns, top_idx)

        if cue.shape[0] == 1 and selected.dim() == 2:
            retrieved = (weights.unsqueeze(-1) * selected).sum(dim=-2)
        else:
            retrieved = (weights.unsqueeze(-1) * selected).sum(dim=-2)

        gate_input = torch.cat([cue, retrieved], dim=-1)
        gate = torch.sigmoid(self.selective_gate(gate_input))
        return self.readout_proj(torch.tanh(gate * retrieved))

    def ranked_selective_recall(self, cue: torch.Tensor, graph) -> torch.Tensor:
        """
        PageRank-enhanced selective recall.

        1차: cosine 유사도로 2×top_k 후보 추출
        2차: MemoryGraph의 PageRank로 재정렬
        3차: 최종 top_k 가중 합산

        Args:
            cue: [batch, d_state]
            graph: MemoryGraph (memory_rank_adapter)
        """
        n_valid = min(self.n_stored.item(), self.capacity)
        if n_valid == 0 or graph is None:
            return self.selective_recall(cue)

        valid_patterns = self.pattern_store[:n_valid]
        cue_norm = F.normalize(cue, dim=-1)
        pattern_norm = F.normalize(valid_patterns, dim=-1)
        similarity = cue_norm @ pattern_norm.T

        # 2× 후보 추출
        n_candidates = min(self.top_k * 2, n_valid)
        top_vals, top_idx = similarity.topk(n_candidates, dim=-1)

        # 첫 번째 배치만으로 reranking (batch-agnostic)
        first_vals = top_vals[0] if top_vals.dim() > 1 else top_vals
        first_idx = top_idx[0] if top_idx.dim() > 1 else top_idx

        candidate_ids = [f"heb_{i.item()}" for i in first_idx]
        sim_scores = first_vals.tolist()

        reranked = graph.rerank_candidates(candidate_ids, sim_scores)

        for cid, _ in reranked[:self.top_k]:
            graph.bump_frequency(cid)

        # 재정렬된 순서로 top_k 패턴 선택
        final_ids = [r[0] for r in reranked[:self.top_k]]
        final_scores = torch.tensor(
            [r[1] for r in reranked[:self.top_k]],
            device=cue.device, dtype=cue.dtype,
        )
        final_weights = F.softmax(final_scores * 5.0, dim=-1)

        final_indices = [int(fid.split("_")[1]) for fid in final_ids]
        selected = valid_patterns[final_indices]

        retrieved = (final_weights.unsqueeze(-1) * selected).sum(dim=0, keepdim=True)
        if cue.shape[0] > 1:
            retrieved = retrieved.expand(cue.shape[0], -1)

        gate_input = torch.cat([cue, retrieved], dim=-1)
        gate = torch.sigmoid(self.selective_gate(gate_input))
        return self.readout_proj(torch.tanh(gate * retrieved))

    def get_pattern_dict(self) -> Dict[str, torch.Tensor]:
        """저장된 패턴을 {node_id: tensor} 형태로 반환."""
        n_valid = min(self.n_stored.item(), self.capacity)
        return {
            f"heb_{i}": self.pattern_store[i]
            for i in range(n_valid)
        }

    def _batch_select(self, patterns: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        B, K = indices.shape
        flat_idx = indices.reshape(-1)
        selected = patterns[flat_idx].reshape(B, K, -1)
        return selected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Episodic Memory (Phase C: 신규)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EpisodicMemory(nn.Module):
    """
    시퀀스 단위 에피소드 기억.

    전체 시퀀스의 상태 궤적을 하나의 '에피소드'로 압축·저장한다.
    새 시퀀스가 들어올 때, 과거 에피소드 중 유사한 것을 찾아 맥락 복원에 활용.

    구조:
    - episode_keys: 에피소드 요약 벡터 (조회용)
    - episode_values: 에피소드 상세 상태 (복원용)
    - 새 에피소드 저장: 시퀀스 끝에서 상태 평균을 key, 최종 상태를 value로 저장
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.n_episodes = cfg.n_episodes
        self.d_state = cfg.d_state

        self.register_buffer(
            "episode_keys", torch.zeros(cfg.n_episodes, cfg.d_state)
        )
        self.register_buffer(
            "episode_values", torch.zeros(cfg.n_episodes, cfg.d_state)
        )
        self.register_buffer("ep_ptr", torch.tensor(0))
        self.register_buffer("n_stored", torch.tensor(0))

        self.query_proj = nn.Linear(cfg.d_state, cfg.d_state, bias=False)
        self.value_proj = nn.Linear(cfg.d_state, cfg.d_state, bias=False)
        self.gate = nn.Linear(cfg.d_state * 2, cfg.d_state)

    def store_episode(self, state_trajectory: torch.Tensor) -> None:
        """
        에피소드 저장.

        Args:
            state_trajectory: [seq_len, d_state] 또는 [batch, seq_len, d_state]
                             (batch면 각 시퀀스를 개별 에피소드로)
        """
        with torch.no_grad():
            if state_trajectory.dim() == 3:
                for b in range(state_trajectory.shape[0]):
                    self._store_one(state_trajectory[b])
            else:
                self._store_one(state_trajectory)

    def _store_one(self, traj: torch.Tensor) -> None:
        key = traj.mean(dim=0)
        value = traj[-1]

        key = F.normalize(key, dim=0)
        value = F.normalize(value, dim=0)

        ptr = self.ep_ptr.item() % self.n_episodes
        self.episode_keys[ptr] = key
        self.episode_values[ptr] = value
        self.ep_ptr.add_(1)
        self.n_stored.add_(1)

    def recall_episode(self, query: torch.Tensor, top_k: int = 2) -> torch.Tensor:
        """
        현재 상태와 유사한 과거 에피소드를 찾아 맥락 신호로 반환.

        Args:
            query: [batch, d_state]
            top_k: 회수할 에피소드 수
        Returns:
            episode_context: [batch, d_state]
        """
        n_valid = min(self.n_stored.item(), self.n_episodes)
        if n_valid == 0:
            return self.value_proj(torch.zeros_like(query))

        q = self.query_proj(query)
        q_norm = F.normalize(q, dim=-1)
        keys_norm = F.normalize(self.episode_keys[:n_valid], dim=-1)

        similarity = q_norm @ keys_norm.T
        k = min(top_k, n_valid)
        top_vals, top_idx = similarity.topk(k, dim=-1)

        weights = F.softmax(top_vals * 5.0, dim=-1)

        values = self.episode_values[:n_valid]
        if query.shape[0] == 1:
            selected = values[top_idx.squeeze(0)]
        else:
            flat = top_idx.reshape(-1)
            selected = values[flat].reshape(query.shape[0], k, -1)

        retrieved = (weights.unsqueeze(-1) * selected).sum(dim=-2)
        projected = self.value_proj(retrieved)

        gate_input = torch.cat([query, projected], dim=-1)
        gate_val = torch.sigmoid(self.gate(gate_input))
        return gate_val * projected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  통합 메모리 시스템 (Phase C)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MemorySystem(nn.Module):
    """
    4-tier 통합 메모리 시스템.

    Phase C: episodic memory 추가, selective recall, WM feedback.
    Phase G: MemoryGraph PageRank 재정렬 옵션.
    """

    def __init__(self, cfg: MemoryConfig, graph=None):
        super().__init__()
        self.cfg = cfg
        self.working = WorkingMemory(cfg)
        self.hebbian = HebbianMemory(cfg)
        self.episodic = EpisodicMemory(cfg)
        self.graph = graph

        self.gate = nn.Linear(cfg.d_state * 3, cfg.d_state)

    def init_state(self, batch_size: int, device: torch.device):
        return self.working.init_state(batch_size, device)

    def step(
        self,
        x: torch.Tensor,
        wm_slots: torch.Tensor,
        wm_activations: torch.Tensor,
        store_to_hebbian: bool = False,
        use_selective: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """메모리 한 스텝."""
        wm_read = self.working.read(x, wm_slots, wm_activations)

        if use_selective and self.hebbian.n_stored.item() > 0:
            if self.graph is not None and self.graph.n_nodes > 0:
                heb_read = self.hebbian.ranked_selective_recall(x, self.graph)
            else:
                heb_read = self.hebbian.selective_recall(x)
        else:
            heb_read = self.hebbian.recall(x)

        ep_read = self.episodic.recall_episode(x)

        combined = torch.cat([wm_read, heb_read, ep_read], dim=-1)
        memory_readout = torch.tanh(self.gate(combined))

        new_slots, new_acts = self.working.write(x, wm_slots, wm_activations)

        if store_to_hebbian:
            self.hebbian.store(x)
            if self.graph is not None:
                ptr = (self.hebbian.store_ptr.item() - 1) % self.hebbian.capacity
                node_id = f"heb_{ptr}"
                existing = self.hebbian.get_pattern_dict()
                self.graph.register_pattern(node_id, x[0].detach(), existing)

        return memory_readout, new_slots, new_acts

    def wm_feedback(self, wm_slots: torch.Tensor, wm_activations: torch.Tensor) -> torch.Tensor:
        """Working memory 요약 → dynamics core 피드백용."""
        return self.working.summary(wm_slots, wm_activations)

    def store_episode(self, state_trajectory: torch.Tensor) -> None:
        """시퀀스 끝에서 에피소드 저장."""
        self.episodic.store_episode(state_trajectory)
