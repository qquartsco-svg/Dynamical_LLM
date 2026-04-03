"""
L2: Dynamics Core — 학습 가능한 벡터장 + ODE 적분

★ 이 파일이 Dynamical LLM Foundation의 핵심이다 ★

핵심 수식:
  ẋ = −λx + F(x, u, m, c; θ)

Phase B 추가:
  1. Context Coupling: 과거 상태 슬라이딩 윈도우로 토큰 간 상호작용
  2. Time-scale Separation: fast/slow 분리 (τ_fast << τ_slow)
  3. Refined Gating: 다중 게이트 (input/state/memory/context 독립) + 잔차 게이트
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .integrator import rk4_step, euler_step
from .stability import StateNorm, clamp_state


@dataclass
class DynamicsConfig:
    d_state: int = 128
    d_input: int = 128
    d_memory: int = 128
    n_substeps: int = 4
    dt: float = 0.25
    damping: float = 0.1
    learnable_damping: bool = True
    use_rk4: bool = True
    max_state_norm: float = 10.0
    coupling_rank: int = 32

    # Phase B
    context_window: int = 16        # 과거 토큰 몇 개를 맥락으로 보는가
    context_heads: int = 4          # 맥락 결합 헤드 수
    use_timescale_sep: bool = True  # fast/slow dynamics 분리
    fast_ratio: float = 0.5         # 상태의 몇 %가 fast track인가
    tau_fast: float = 0.1           # fast dynamics 시간상수
    tau_slow: float = 1.0           # slow dynamics 시간상수


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Context Coupling — 토큰 간 상호작용
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ContextCoupling(nn.Module):
    """
    과거 상태들과의 결합을 통해 '맥락 선택' 기능을 구현한다.

    Transformer의 attention과 목적은 같지만 구조가 다르다:
    - attention: Q·K^T → softmax → V (O(n²))
    - 이것: 현재 상태와 과거 윈도우 사이의 선형 결합 + 게이팅 (O(n·w))

    Multi-head 구조로 서로 다른 시간 범위의 맥락을 동시에 본다.
    """

    def __init__(self, d_state: int, n_heads: int = 4, window: int = 16):
        super().__init__()
        self.d_state = d_state
        self.n_heads = n_heads
        self.window = window
        self.d_head = d_state // n_heads

        self.q_proj = nn.Linear(d_state, d_state, bias=False)
        self.k_proj = nn.Linear(d_state, d_state, bias=False)
        self.v_proj = nn.Linear(d_state, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_state, bias=False)

        self.gate = nn.Linear(d_state * 2, d_state)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, d_state] 현재 상태
            history: [batch, window, d_state] 과거 상태 윈도우
        Returns:
            context: [batch, d_state] 맥락 결합 결과
        """
        B, W, D = history.shape

        q = self.q_proj(x).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(history).view(B, W, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(history).view(B, W, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        scores = (q @ k.transpose(-2, -1)) / scale
        weights = F.softmax(scores, dim=-1)

        context = (weights @ v).transpose(1, 2).reshape(B, D)
        context = self.out_proj(context)

        gate_val = torch.sigmoid(self.gate(torch.cat([x, context], dim=-1)))
        return gate_val * context


class ContextBuffer:
    """
    과거 상태를 슬라이딩 윈도우로 유지.
    """

    def __init__(self, window: int, d_state: int):
        self.window = window
        self.d_state = d_state
        self._buf: torch.Tensor | None = None

    def init(self, batch_size: int, device: torch.device):
        self._buf = torch.zeros(batch_size, self.window, self.d_state, device=device)

    def push(self, x: torch.Tensor):
        """새 상태를 버퍼 끝에 추가, 가장 오래된 상태 제거."""
        self._buf = torch.cat([self._buf[:, 1:, :], x.unsqueeze(1).detach()], dim=1)

    def get(self) -> torch.Tensor:
        return self._buf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Time-Scale Separation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TimescaleSeparator(nn.Module):
    """
    상태벡터를 fast / slow 트랙으로 분리.

    fast: 현재 입력에 빠르게 반응 (τ 작음)
    slow: 장기 맥락을 천천히 축적 (τ 큼)

    최종 출력은 두 트랙을 게이트로 합친다.
    생물학적으로 대뇌피질의 multi-timescale 구조에 대응.
    """

    def __init__(self, d_state: int, fast_ratio: float = 0.5):
        super().__init__()
        self.d_fast = int(d_state * fast_ratio)
        self.d_slow = d_state - self.d_fast

        self.merge_gate = nn.Linear(d_state, d_state)
        nn.init.xavier_uniform_(self.merge_gate.weight, gain=0.1)
        nn.init.zeros_(self.merge_gate.bias)

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., :self.d_fast], x[..., self.d_fast:]

    def merge(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([fast, slow], dim=-1)
        gate = torch.sigmoid(self.merge_gate(combined))
        return gate * combined


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VectorField (Phase B 확장)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VectorField(nn.Module):
    """
    학습 가능한 벡터장 F(x, u, m, c; θ).

    Phase B:
      F = residual_gate ⊙ x + gate ⊙ σ(state + input + memory + context + b)
      - 잔차 게이트: 얼마나 이전 상태를 유지할지
      - 독립 게이트: input/memory/context 각각에 별도 게이트
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        d = cfg.d_state
        r = cfg.coupling_rank

        # State coupling: low-rank
        self.W_down = nn.Linear(d, r, bias=False)
        self.W_up = nn.Linear(r, d, bias=False)

        # Input coupling
        self.W_input = nn.Linear(cfg.d_input, d, bias=False)

        # Memory coupling
        self.W_memory = nn.Linear(cfg.d_memory, d, bias=False)

        # Context coupling projection
        self.W_context = nn.Linear(d, d, bias=False)

        # Independent gates
        self.gate_input = nn.Linear(d + cfg.d_input, d)
        self.gate_memory = nn.Linear(d + cfg.d_memory, d)
        self.gate_context = nn.Linear(d * 2, d)

        # Residual gate: 이전 상태 보존 비율
        self.residual_gate = nn.Linear(d + cfg.d_input, d)

        self.bias = nn.Parameter(torch.zeros(d))
        self.norm = StateNorm(d)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_down, self.W_up, self.W_input, self.W_memory, self.W_context]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        for m in [self.gate_input, self.gate_memory, self.gate_context, self.residual_gate]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        m: torch.Tensor,
        c: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        F(x, u, m, c; θ).

        Args:
            x: [batch, d_state]
            u: [batch, d_input]
            m: [batch, d_memory]
            c: [batch, d_state] context coupling (None이면 0)
        Returns:
            dx: [batch, d_state]
        """
        # State coupling (low-rank)
        state_coupling = self.W_up(torch.tanh(self.W_down(x)))

        # Input: gated
        input_raw = self.W_input(u)
        g_in = torch.sigmoid(self.gate_input(torch.cat([x, u], dim=-1)))
        input_coupling = g_in * input_raw

        # Memory: gated
        memory_raw = self.W_memory(m)
        g_mem = torch.sigmoid(self.gate_memory(torch.cat([x, m], dim=-1)))
        memory_coupling = g_mem * memory_raw

        # Context: gated
        if c is not None:
            context_raw = self.W_context(c)
            g_ctx = torch.sigmoid(self.gate_context(torch.cat([x, c], dim=-1)))
            context_coupling = g_ctx * context_raw
        else:
            context_coupling = torch.zeros_like(x)

        pre = state_coupling + input_coupling + memory_coupling + context_coupling + self.bias

        # Residual gate
        r = torch.sigmoid(self.residual_gate(torch.cat([x, u], dim=-1)))

        return r * x + (1 - r) * torch.tanh(pre)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DynamicsCore (Phase B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DynamicsCore(nn.Module):
    """
    동역학 코어.

    Phase B:
    - ContextCoupling + ContextBuffer로 토큰 간 상호작용
    - TimescaleSeparator로 fast/slow 분리
    - 정제된 VectorField
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.field = VectorField(cfg)
        self.output_norm = StateNorm(cfg.d_state)

        self.context_coupling = ContextCoupling(
            cfg.d_state, n_heads=cfg.context_heads, window=cfg.context_window
        )

        self.timescale = None
        if cfg.use_timescale_sep:
            self.timescale = TimescaleSeparator(cfg.d_state, cfg.fast_ratio)

        if cfg.learnable_damping:
            self._log_damping = nn.Parameter(torch.tensor(math.log(cfg.damping)))
        else:
            self.register_buffer("_log_damping", torch.tensor(math.log(cfg.damping)))

        if cfg.use_timescale_sep:
            self._log_tau_fast = nn.Parameter(torch.tensor(math.log(cfg.tau_fast)))
            self._log_tau_slow = nn.Parameter(torch.tensor(math.log(cfg.tau_slow)))

    @property
    def damping(self) -> torch.Tensor:
        return torch.exp(self._log_damping)

    def vector_field_fn(
        self, x: torch.Tensor, u: torch.Tensor, m: torch.Tensor, c: torch.Tensor
    ):
        """ẋ = −λx + F(x, u, m, c; θ)"""
        lam = self.damping

        if self.timescale is not None:
            fast, slow = self.timescale.split(x)
            tau_f = torch.exp(self._log_tau_fast)
            tau_s = torch.exp(self._log_tau_slow)

            f_full = self.field(x, u, m, c)
            f_fast, f_slow = self.timescale.split(f_full)

            dx_fast = (-lam * fast + f_fast) / tau_f
            dx_slow = (-lam * slow + f_slow) / tau_s

            return torch.cat([dx_fast, dx_slow], dim=-1)

        return -lam * x + self.field(x, u, m, c)

    def step_token(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        m: torch.Tensor,
        ctx_buf: ContextBuffer = None,
    ) -> torch.Tensor:
        """
        단일 토큰에 대한 상태 진화.

        Args:
            x: [batch, d_state]
            u: [batch, d_input]
            m: [batch, d_memory]
            ctx_buf: 과거 상태 버퍼 (None이면 context coupling 비활성)
        """
        c = torch.zeros_like(x)
        if ctx_buf is not None:
            history = ctx_buf.get()
            c = self.context_coupling(x, history)

        dt = self.cfg.dt
        step_fn = rk4_step if self.cfg.use_rk4 else euler_step

        def f(state, t):
            return self.vector_field_fn(state, u, m, c)

        t = torch.zeros(1, device=x.device)
        for _ in range(self.cfg.n_substeps):
            x = step_fn(f, x, t, dt)
            x = clamp_state(x, self.cfg.max_state_norm)
            t = t + dt

        x = self.output_norm(x)

        if ctx_buf is not None:
            ctx_buf.push(x)

        return x

    def forward(
        self,
        x0: torch.Tensor,
        impulses: torch.Tensor,
        memory_readouts: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        시퀀스 전체에 대한 상태 진화.

        Args:
            x0: [batch, d_state]
            impulses: [batch, seq_len, d_input]
            memory_readouts: [batch, seq_len, d_memory] 또는 None
        """
        batch, seq_len, _ = impulses.shape
        device = impulses.device

        if memory_readouts is None:
            memory_readouts = torch.zeros(
                batch, seq_len, self.cfg.d_memory, device=device
            )

        ctx_buf = ContextBuffer(self.cfg.context_window, self.cfg.d_state)
        ctx_buf.init(batch, device)

        states = []
        x = x0

        for t in range(seq_len):
            u = impulses[:, t, :]
            m = memory_readouts[:, t, :]
            x = self.step_token(x, u, m, ctx_buf)
            states.append(x)

        return torch.stack(states, dim=1)
