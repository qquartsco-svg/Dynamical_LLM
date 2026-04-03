"""
L4: Readout — 상태 → 어휘 확률
L5: Online Adaptation — 추론 중 미세 적응

Phase D 고도화:
  - OnlineAdapter: dynamics 상태에도 적응 가능 (readout만이 아님)
  - fast weight decay: 시간에 따라 잊어가는 fast weights
  - ConsolidationScheduler: fast → slow 주기적 통합 자동화
  - 적응 이력 기록 (진단용)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stability import TrustGate


class Readout(nn.Module):
    """L4: 상태 → 어휘 logits."""

    def __init__(self, d_state: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_state, vocab_size)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.proj(states)


class OnlineAdapter(nn.Module):
    """
    L5: Cerebellum 패턴 온라인 적응.

    Phase D 고도화:
    - fast_weight_decay: 매 adapt마다 이전 fast weights를 감쇠
      → 오래된 적응이 자연스럽게 사라짐
    - adapt_state(): readout뿐 아니라 상태 변환에도 적응 가능 (선택)
    - 적응 이력: n_updates, cumulative trust, loss trajectory
    """

    def __init__(self, d_state: int, vocab_size: int, lr: float = 0.001,
                 fast_decay: float = 0.999):
        super().__init__()
        self.d_state = d_state
        self.vocab_size = vocab_size
        self.lr = lr
        self.fast_decay = fast_decay
        self.trust_gate = TrustGate()

        self.register_buffer("fast_weight", torch.zeros(vocab_size, d_state))
        self.register_buffer("fast_bias", torch.zeros(vocab_size))
        self.register_buffer("n_updates", torch.tensor(0))
        self.register_buffer("cumulative_trust", torch.tensor(0.0))

        self.register_buffer(
            "state_adapter", torch.zeros(d_state, d_state)
        )

    def adapt(
        self,
        state: torch.Tensor,
        target_id: int,
        logits: torch.Tensor,
    ) -> dict:
        """
        한 스텝 온라인 적응 + 진단 정보 반환.

        Returns:
            dict: trust, loss, n_updates
        """
        with torch.no_grad():
            self.fast_weight.mul_(self.fast_decay)
            self.fast_bias.mul_(self.fast_decay)

            probs = F.softmax(logits, dim=-1)

            target_vec = torch.zeros(self.vocab_size, device=state.device)
            target_vec[target_id] = 1.0
            error = target_vec - probs

            loss = -torch.log(probs[target_id] + 1e-8).item()
            trust = self.trust_gate.update(loss)

            update = self.lr * trust * torch.outer(error, state)
            self.fast_weight.add_(update)
            self.fast_bias.add_(self.lr * trust * error)
            self.n_updates.add_(1)
            self.cumulative_trust.add_(trust)

        return {"trust": trust, "loss": loss, "n_updates": self.n_updates.item()}

    def adapt_state(self, state: torch.Tensor, target_state: torch.Tensor) -> None:
        """
        상태 변환 적응: 현재 상태에서 목표 상태로의 매핑을 fast weight로 학습.
        dynamics core의 미세 조정에 활용.
        """
        with torch.no_grad():
            trust = self.trust_gate.trust
            error = target_state - state
            update = self.lr * trust * 0.1 * torch.outer(error.squeeze(0), state.squeeze(0))
            self.state_adapter.add_(update)
            self.state_adapter.mul_(self.fast_decay)

    def apply_fast(self, logits: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """fast weights를 logits에 적용."""
        fast_logits = F.linear(states, self.fast_weight, self.fast_bias)
        return logits + fast_logits

    def apply_state_adapter(self, state: torch.Tensor) -> torch.Tensor:
        """state adapter를 현재 상태에 적용 (잔차 형태)."""
        return state + F.linear(state, self.state_adapter) * 0.1

    def consolidate(self, readout: Readout, ratio: float = 0.01) -> None:
        """fast weights를 slow weights로 통합."""
        with torch.no_grad():
            readout.proj.weight.add_(ratio * self.fast_weight)
            readout.proj.bias.add_(ratio * self.fast_bias)
            self.fast_weight.mul_(1.0 - ratio)
            self.fast_bias.mul_(1.0 - ratio)

    def reset(self) -> None:
        self.fast_weight.zero_()
        self.fast_bias.zero_()
        self.state_adapter.zero_()
        self.n_updates.zero_()
        self.cumulative_trust.zero_()

    @property
    def adaptation_summary(self) -> dict:
        return {
            "n_updates": self.n_updates.item(),
            "avg_trust": (self.cumulative_trust / max(self.n_updates, 1)).item(),
            "fast_weight_norm": self.fast_weight.norm().item(),
            "state_adapter_norm": self.state_adapter.norm().item(),
        }


class ConsolidationScheduler:
    """
    Fast → Slow 통합 스케줄러.

    일정 간격마다 자동으로 consolidation을 수행한다.
    trust가 높으면 더 적극적으로 통합, 낮으면 보수적으로.
    """

    def __init__(
        self,
        interval: int = 100,
        base_ratio: float = 0.01,
        trust_scale: bool = True,
    ):
        self.interval = interval
        self.base_ratio = base_ratio
        self.trust_scale = trust_scale
        self._step_count = 0
        self._consolidation_count = 0

    def step(self, adapter: OnlineAdapter, readout: Readout) -> bool:
        """
        한 스텝. interval마다 자동 consolidation 수행.

        Returns:
            True if consolidation 수행됨
        """
        self._step_count += 1
        if self._step_count % self.interval != 0:
            return False

        ratio = self.base_ratio
        if self.trust_scale:
            ratio *= adapter.trust_gate.trust

        if adapter.n_updates.item() > 0:
            adapter.consolidate(readout, ratio=ratio)
            self._consolidation_count += 1
            return True
        return False

    @property
    def stats(self) -> dict:
        return {
            "total_steps": self._step_count,
            "consolidations": self._consolidation_count,
        }
