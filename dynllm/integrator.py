"""
RK4 적분기 — 동역학 LLM의 시간 진화 엔진

기존 00_BRAIN ODE 엔진들(KEMET, VectorSpace 등)의
RK4 패턴을 PyTorch 텐서 연산에 맞게 구현한다.

dx/dt = f(x, t) → x_{t+dt} = RK4(f, x, t, dt)
"""
from __future__ import annotations

from typing import Callable

import torch


def rk4_step(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    단일 RK4 스텝.

    Args:
        f: vector field  f(x, t) → dx/dt
        x: 현재 상태 [batch, d_state]
        t: 현재 시간 [scalar or batch]
        dt: 시간 스텝
    Returns:
        x_next: 다음 상태 [batch, d_state]
    """
    dt_half = dt * 0.5
    t_half = t + dt_half
    t_full = t + dt

    k1 = f(x, t)
    k2 = f(x + dt_half * k1, t_half)
    k3 = f(x + dt_half * k2, t_half)
    k4 = f(x + dt * k3, t_full)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """단일 Euler 스텝 (빠르지만 정확도 낮음)."""
    return x + dt * f(x, t)


def adaptive_step(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    tol: float = 1e-4,
) -> tuple[torch.Tensor, float]:
    """
    적응형 스텝 사이즈. 오차 추정으로 dt를 조절한다.

    Returns:
        x_next, dt_next
    """
    x_full = rk4_step(f, x, t, dt)
    x_half1 = rk4_step(f, x, t, dt * 0.5)
    x_half2 = rk4_step(f, x_half1, t + dt * 0.5, dt * 0.5)

    err = torch.max(torch.abs(x_full - x_half2)).item()

    if err < tol * 0.1:
        dt_next = min(dt * 2.0, 1.0)
    elif err > tol:
        dt_next = max(dt * 0.5, 1e-6)
    else:
        dt_next = dt

    return x_half2, dt_next
