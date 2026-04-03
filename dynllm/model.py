"""
DynLLM — 동역학 LLM 통합 모델

Phase C+D:
  - Episodic memory: 시퀀스 끝에서 에피소드 저장
  - Selective recall: Hebbian에서 관련 패턴만 선택 회수
  - WM feedback: 작업기억 요약이 dynamics에 재주입
  - RollbackPolicy: 이력 기반 자동 복원
  - ConsolidationScheduler: fast→slow 자동 통합

Phase G: 엔진 허브 통합
  - MemoryGraph: PageRank 기반 기억 재정렬
  - DynLLMDiagnostics: 수렴/엔트로피/통합정보 진단
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics_core import DynamicsConfig, DynamicsCore, ContextBuffer
from .memory import MemoryConfig, MemorySystem
from .readout import Readout, OnlineAdapter, ConsolidationScheduler
from .state_encoder import StateEncoder
from .stability import detect_instability, SafeStateBuffer, RollbackPolicy


@dataclass
class DynLLMConfig:
    vocab_size: int = 256
    d_state: int = 128
    n_substeps: int = 4
    dt: float = 0.25
    damping: float = 0.1
    coupling_rank: int = 32
    n_working_slots: int = 8
    hebbian_size: int = 256
    use_memory: bool = True
    use_adaptation: bool = False

    # Phase B
    context_window: int = 16
    context_heads: int = 4
    use_timescale_sep: bool = True
    fast_ratio: float = 0.5
    tau_fast: float = 0.1
    tau_slow: float = 1.0

    # Phase C
    n_episodes: int = 16
    episode_len: int = 32
    selective_top_k: int = 3

    # Phase G: Engine Hub 통합
    use_memory_rank: bool = False
    use_diagnostics: bool = False

    @property
    def dynamics_config(self) -> DynamicsConfig:
        return DynamicsConfig(
            d_state=self.d_state,
            d_input=self.d_state,
            d_memory=self.d_state,
            n_substeps=self.n_substeps,
            dt=self.dt,
            damping=self.damping,
            coupling_rank=self.coupling_rank,
            context_window=self.context_window,
            context_heads=self.context_heads,
            use_timescale_sep=self.use_timescale_sep,
            fast_ratio=self.fast_ratio,
            tau_fast=self.tau_fast,
            tau_slow=self.tau_slow,
        )

    @property
    def memory_config(self) -> MemoryConfig:
        return MemoryConfig(
            d_state=self.d_state,
            n_working_slots=self.n_working_slots,
            hebbian_size=self.hebbian_size,
            n_episodes=self.n_episodes,
            episode_len=self.episode_len,
            selective_top_k=self.selective_top_k,
        )


class DynLLM(nn.Module):
    """
    Dynamical LLM Foundation — 전체 모델.

    ẋ = −λx + F(x, u, m, c; θ)
    """

    def __init__(self, cfg: DynLLMConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = StateEncoder(cfg.vocab_size, cfg.d_state)
        self.dynamics = DynamicsCore(cfg.dynamics_config)

        # Phase G: MemoryGraph
        self._memory_graph = None
        if cfg.use_memory_rank and cfg.use_memory:
            from .memory_rank_adapter import MemoryGraph
            self._memory_graph = MemoryGraph()

        self.memory = MemorySystem(cfg.memory_config, graph=self._memory_graph) if cfg.use_memory else None
        self.readout = Readout(cfg.d_state, cfg.vocab_size)
        self.adapter = OnlineAdapter(cfg.d_state, cfg.vocab_size) if cfg.use_adaptation else None

        # Phase G: Diagnostics
        self._diagnostics = None
        if cfg.use_diagnostics:
            from .diagnostics import DynLLMDiagnostics
            self._diagnostics = DynLLMDiagnostics()

    def forward(
        self,
        input_ids: torch.Tensor,
        x0: torch.Tensor = None,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        device = input_ids.device

        impulses = self.encoder(input_ids)

        if x0 is None:
            x0 = torch.zeros(batch, self.cfg.d_state, device=device)

        if self.memory is not None:
            states = self._forward_with_memory(x0, impulses, batch, seq_len, device)
        else:
            states = self.dynamics(x0, impulses)

        logits = self.readout(states)

        if self.adapter is not None:
            logits = self.adapter.apply_fast(logits, states)

        return logits

    def _forward_with_memory(
        self,
        x0: torch.Tensor,
        impulses: torch.Tensor,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """메모리 + 맥락 결합 + 에피소드 순전파."""
        wm_slots, wm_acts = self.memory.init_state(batch, device)

        ctx_buf = ContextBuffer(self.cfg.context_window, self.cfg.d_state)
        ctx_buf.init(batch, device)

        states = []
        x = x0

        for t in range(seq_len):
            u = impulses[:, t, :]

            mem_read, wm_slots, wm_acts = self.memory.step(
                x, wm_slots, wm_acts,
                store_to_hebbian=(t % 4 == 0),
                use_selective=True,
            )

            x = self.dynamics.step_token(x, u, mem_read, ctx_buf)
            states.append(x)

        state_tensor = torch.stack(states, dim=1)

        if self.training:
            self.memory.store_episode(state_tensor.detach())

        return state_tensor

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 0,
    ) -> list[int]:
        """자기회귀 생성 + RollbackPolicy."""
        self.eval()
        device = next(self.parameters()).device

        ids = list(prompt_ids)
        x = torch.zeros(1, self.cfg.d_state, device=device)
        safe_buf = SafeStateBuffer(max_history=5)
        rollback_policy = RollbackPolicy()

        ctx_buf = ContextBuffer(self.cfg.context_window, self.cfg.d_state)
        ctx_buf.init(1, device)

        wm_slots, wm_acts = None, None
        if self.memory:
            wm_slots, wm_acts = self.memory.init_state(1, device)

        for token_id in ids:
            u = self.encoder.encode_single(token_id).unsqueeze(0).to(device)
            if self.memory and wm_slots is not None:
                mem_read, wm_slots, wm_acts = self.memory.step(x, wm_slots, wm_acts)
                x = self.dynamics.step_token(x, u, mem_read, ctx_buf)
            else:
                m = torch.zeros(1, self.cfg.d_state, device=device)
                x = self.dynamics.step_token(x, u, m, ctx_buf)
            safe_buf.save(x)

        generated = []
        for _ in range(max_tokens):
            logits = self.readout(x).squeeze(0)

            if self.adapter:
                fast_logits = F.linear(x.squeeze(0), self.adapter.fast_weight, self.adapter.fast_bias)
                logits = logits + fast_logits

            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == 2:  # EOS
                break

            generated.append(next_id)

            u = self.encoder.encode_single(next_id).unsqueeze(0).to(device)
            if self.memory and wm_slots is not None:
                mem_read, wm_slots, wm_acts = self.memory.step(x, wm_slots, wm_acts)
                x = self.dynamics.step_token(x, u, mem_read, ctx_buf)
            else:
                m = torch.zeros(1, self.cfg.d_state, device=device)
                x = self.dynamics.step_token(x, u, m, ctx_buf)

            x, _ = rollback_policy.check_and_rollback(x, safe_buf)

        return generated

    def run_diagnostics(self, loss: Optional[float] = None, logits: Optional[torch.Tensor] = None) -> dict:
        """
        Phase G: 엔진 허브 통합 진단.

        학습 중 loss와 logits를 기록하면 수렴/엔트로피/통합정보를 추적한다.
        """
        if self._diagnostics is None:
            return {"diagnostics": "disabled"}

        if loss is not None:
            self._diagnostics.record_loss(loss)

        if logits is not None:
            self._diagnostics.record_logits(logits)

        coupling_matrix = None
        try:
            vf = self.dynamics.field
            if hasattr(vf, "state_coupling"):
                W = vf.state_coupling
                if hasattr(W, "weight"):
                    coupling_matrix = W.weight.detach()
        except Exception:
            pass

        return self._diagnostics.full_diagnostic(coupling_matrix)

    @property
    def memory_graph(self):
        """MemoryGraph 인스턴스 (없으면 None)."""
        return self._memory_graph

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.cfg,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DynLLM":
        data = torch.load(path, map_location=device, weights_only=False)
        model = cls(data["config"])
        model.load_state_dict(data["state_dict"])
        return model
