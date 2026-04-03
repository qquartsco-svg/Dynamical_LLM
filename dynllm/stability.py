"""
안정성 제어 — 동역학 LLM의 발산/붕괴 방지

Phase C+D 추가:
  - TrustGate: 학습 가능한 파라미터 + EMA 기반 자동 스케줄
  - RollbackPolicy: 이력 기반 자동 복원 전략 (연속 불안정 감지, cooldown)
  - SafeStateBuffer: 다단계 이력 + 보간 복원
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class StateNorm(nn.Module):
    """상태벡터 정규화."""

    def __init__(self, d_state: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_state, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def clamp_state(x: torch.Tensor, max_norm: float = 10.0) -> torch.Tensor:
    """상태벡터 크기 제한."""
    norms = x.norm(dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
    return x * scale


def detect_instability(x: torch.Tensor, threshold: float = 100.0) -> bool:
    """불안정 감지."""
    return bool(x.abs().max().item() > threshold or torch.isnan(x).any().item())


class SafeStateBuffer:
    """
    안전한 이전 상태 보관 + 다단계 복원.

    Phase D: 보간 복원 — 최근 2개 안전 상태의 중간값으로도 복원 가능.
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self._history: list[torch.Tensor] = []
        self._rollback_count = 0

    def save(self, x: torch.Tensor) -> None:
        if not detect_instability(x):
            self._history.append(x.detach().clone())
            if len(self._history) > self.max_history:
                self._history.pop(0)

    def rollback(self, x: torch.Tensor) -> torch.Tensor:
        """최근 안전 상태로 복원."""
        self._rollback_count += 1
        if self._history:
            return self._history[-1].to(x.device)
        return x * 0.01

    def interpolated_rollback(self, x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """최근 2개 안전 상태의 보간으로 복원."""
        self._rollback_count += 1
        if len(self._history) >= 2:
            s1 = self._history[-1].to(x.device)
            s2 = self._history[-2].to(x.device)
            return alpha * s1 + (1 - alpha) * s2
        return self.rollback(x)

    @property
    def has_history(self) -> bool:
        return len(self._history) > 0

    @property
    def rollback_count(self) -> int:
        return self._rollback_count


class VelocityMonitor:
    """상태 변화 속도(dx/dt) 추적."""

    def __init__(self, window: int = 20, accel_threshold: float = 5.0):
        self.window = window
        self.accel_threshold = accel_threshold
        self._velocities: list[float] = []

    def record(self, x_prev: torch.Tensor, x_next: torch.Tensor, dt: float = 1.0) -> dict:
        velocity = (x_next - x_prev).norm().item() / max(dt, 1e-8)
        self._velocities.append(velocity)
        if len(self._velocities) > self.window:
            self._velocities.pop(0)

        accel = 0.0
        if len(self._velocities) >= 2:
            accel = self._velocities[-1] - self._velocities[-2]

        return {
            "velocity": velocity,
            "acceleration": accel,
            "warning": abs(accel) > self.accel_threshold,
            "avg_velocity": sum(self._velocities) / len(self._velocities),
        }


def apply_spectral_norm(module: nn.Module, targets: tuple[type, ...] = (nn.Linear,)) -> nn.Module:
    """모듈 내 지정 타입 레이어에 spectral normalization 적용."""
    for name, child in module.named_children():
        if isinstance(child, targets) and not hasattr(child, "weight_orig"):
            setattr(module, name, spectral_norm(child))
        else:
            apply_spectral_norm(child, targets)
    return module


class TrustGate(nn.Module):
    """
    적응 신뢰도 게이트.

    Phase D 고도화:
    - EMA 기반 손실 추적 (window보다 더 안정적)
    - 학습 가능한 sensitivity 파라미터
    - 자동 warmup: 초기엔 보수적, 안정되면 신뢰 회복
    """

    def __init__(self, window: int = 50, threshold: float = 2.0, ema_alpha: float = 0.1):
        super().__init__()
        self.window = window
        self.threshold = threshold
        self.ema_alpha = ema_alpha

        self._losses: list[float] = []
        self._trust = 1.0
        self._ema_loss = 0.0
        self._ema_initialized = False
        self._step_count = 0

    def update(self, loss: float) -> float:
        self._step_count += 1
        self._losses.append(loss)
        if len(self._losses) > self.window:
            self._losses.pop(0)

        if not self._ema_initialized:
            self._ema_loss = loss
            self._ema_initialized = True
        else:
            self._ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self._ema_loss

        if len(self._losses) >= 10:
            recent = self._losses[-5:]
            older = self._losses[-10:-5]
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)

            if older_avg > 0 and recent_avg / older_avg > self.threshold:
                self._trust = max(0.1, self._trust * 0.8)
            elif recent_avg < self._ema_loss:
                self._trust = min(1.0, self._trust * 1.02)
            else:
                self._trust = min(1.0, self._trust * 1.005)

        # warmup
        if self._step_count < 20:
            self._trust = min(self._trust, self._step_count / 20.0)

        return self._trust

    @property
    def trust(self) -> float:
        return self._trust

    @property
    def ema_loss(self) -> float:
        return self._ema_loss

    @property
    def diagnostics(self) -> dict:
        return {
            "trust": self._trust,
            "ema_loss": self._ema_loss,
            "step_count": self._step_count,
            "recent_losses": self._losses[-5:] if self._losses else [],
        }


class RollbackPolicy:
    """
    이력 기반 자동 복원 전략.

    연속 불안정 감지 시:
    1. 첫 번째: 마지막 안전 상태로 rollback
    2. 연속 2회: 보간 rollback (더 보수적 복원)
    3. 연속 3회+: cooldown 모드 (damping 강화 신호 전달)

    cooldown이 끝나면 정상 모드로 복귀.
    """

    def __init__(self, consecutive_limit: int = 3, cooldown_steps: int = 10):
        self.consecutive_limit = consecutive_limit
        self.cooldown_steps = cooldown_steps
        self._consecutive_failures = 0
        self._cooldown_remaining = 0
        self._total_rollbacks = 0

    def check_and_rollback(
        self,
        x: torch.Tensor,
        safe_buf: SafeStateBuffer,
    ) -> tuple[torch.Tensor, dict]:
        """
        불안정 감지 + 자동 복원.

        Returns:
            (복원된 상태, 진단 정보)
        """
        info = {"rolled_back": False, "mode": "normal", "cooldown": self._cooldown_remaining}

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            info["mode"] = "cooldown"

        if detect_instability(x):
            self._consecutive_failures += 1
            self._total_rollbacks += 1
            info["rolled_back"] = True

            if self._consecutive_failures >= self.consecutive_limit:
                x = safe_buf.interpolated_rollback(x, alpha=0.3)
                self._cooldown_remaining = self.cooldown_steps
                info["mode"] = "cooldown_triggered"
            elif self._consecutive_failures >= 2:
                x = safe_buf.interpolated_rollback(x, alpha=0.5)
                info["mode"] = "interpolated_rollback"
            else:
                x = safe_buf.rollback(x)
                info["mode"] = "simple_rollback"
        else:
            self._consecutive_failures = 0
            safe_buf.save(x)

        return x, info

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown_remaining > 0

    @property
    def diagnostics(self) -> dict:
        return {
            "consecutive_failures": self._consecutive_failures,
            "cooldown_remaining": self._cooldown_remaining,
            "total_rollbacks": self._total_rollbacks,
        }
