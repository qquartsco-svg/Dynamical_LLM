"""
DynLLM 진단 어댑터 — 00_BRAIN 엔진 허브 연동

3개 진단 엔진을 DynLLM 추론/학습에 연결한다:

  1. ConvergenceMonitor   (← ConvergenceDynamics_Engine)
     학습 loss 시계열 → 수렴 차수, Lyapunov 추정, 안정성 판정

  2. EntropyAnalyzer      (← StatMech_Engine)
     출력 분포 → Gibbs 엔트로피, 다양성, 온도 추정

  3. IntegrationDiagnostic (← IIT_Engine)
     내부 상태 결합 행렬 → Φ 근사 (통합 정보 진단)

각 어댑터는 자체 구현이 기본이며, ENGINE_HUB 엔진이 import 가능하면
실제 엔진으로 위임한다. (독립 실행 원칙)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENGINE_HUB 경로 탐색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BRAIN_ROOT = Path(__file__).resolve().parents[3]
_ENGINE_HUB = _BRAIN_ROOT / "ENGINE_HUB"


def _try_import(engine_path: Path, module_name: str):
    """ENGINE_HUB 경로에서 모듈 import 시도."""
    if engine_path.is_dir():
        parent = str(engine_path)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            return __import__(module_name)
        except ImportError:
            pass
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. ConvergenceMonitor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ConvergenceResult:
    """수렴 분석 결과."""
    convergence_order: float = 0.0
    lyapunov_estimate: float = 0.0
    health: float = 0.0
    stable: bool = True
    verdict: str = ""


class ConvergenceMonitor:
    """
    학습/추론 수렴 안정성 진단.

    ENGINE_HUB 연결: ConvergenceDynamics_Engine (40_SPATIAL_LAYER)
    자체 구현: loss 비율 기반 수렴 차수 + Lyapunov 추정
    """

    def __init__(self, window: int = 50):
        self.window = window
        self._history: List[float] = []
        self._ext_engine = None
        self._try_load_engine()

    def _try_load_engine(self):
        mod = _try_import(
            _ENGINE_HUB / "2_operational" / "40_SPATIAL_LAYER" / "ConvergenceDynamics_Engine",
            "engine",
        )
        if mod and hasattr(mod, "ConvergenceDynamicsEngine"):
            self._ext_engine = mod.ConvergenceDynamicsEngine()

    def record(self, value: float):
        self._history.append(value)
        if len(self._history) > self.window * 3:
            self._history = self._history[-self.window * 2:]

    def analyze(self) -> ConvergenceResult:
        """현재 이력에서 수렴 상태를 분석한다."""
        if self._ext_engine is not None and len(self._history) >= 5:
            return self._analyze_with_engine()
        return self._analyze_builtin()

    def _analyze_with_engine(self) -> ConvergenceResult:
        try:
            result = self._ext_engine.analyze(self._history)
            health = self._ext_engine.health(result)
            return ConvergenceResult(
                convergence_order=getattr(result, "convergence_order", 0.0),
                lyapunov_estimate=getattr(result, "lyapunov", 0.0),
                health=health,
                stable=getattr(result, "stability", "stable") != "diverging",
                verdict=getattr(result, "label", ""),
            )
        except Exception:
            return self._analyze_builtin()

    def _analyze_builtin(self) -> ConvergenceResult:
        h = self._history
        if len(h) < 5:
            return ConvergenceResult(verdict="insufficient_data")

        recent = h[-5:]
        older = h[-10:-5] if len(h) >= 10 else h[:5]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        # Convergence order approximation
        errors = [abs(h[i] - h[i - 1]) for i in range(1, len(h)) if abs(h[i - 1]) > 1e-12]
        conv_order = 0.0
        if len(errors) >= 3:
            ratios = []
            for i in range(1, len(errors)):
                if errors[i - 1] > 1e-12:
                    ratios.append(errors[i] / errors[i - 1])
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                conv_order = -math.log(max(avg_ratio, 1e-12)) if avg_ratio < 1 else 0.0

        # Lyapunov-style estimate
        lyap = 0.0
        if older_avg > 1e-12:
            lyap = math.log(max(recent_avg / older_avg, 1e-12))

        # Stability
        diverging = recent_avg > older_avg * 1.5
        collapsing = recent_avg < 1e-12 and older_avg > 0.01

        health = 1.0
        if diverging:
            health = max(0.0, 1.0 - (recent_avg / max(older_avg, 1e-8) - 1.0))
        elif collapsing:
            health = 0.3

        if diverging:
            verdict = "diverging"
        elif collapsing:
            verdict = "collapsed"
        elif conv_order > 1.0:
            verdict = "converging_fast"
        elif conv_order > 0.1:
            verdict = "converging"
        else:
            verdict = "plateau"

        return ConvergenceResult(
            convergence_order=conv_order,
            lyapunov_estimate=lyap,
            health=min(1.0, max(0.0, health)),
            stable=not diverging,
            verdict=verdict,
        )

    @property
    def history(self) -> List[float]:
        return list(self._history)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. EntropyAnalyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EntropyResult:
    """엔트로피 분석 결과."""
    entropy: float = 0.0
    max_entropy: float = 0.0
    normalized_entropy: float = 0.0
    temperature_estimate: float = 1.0


class EntropyAnalyzer:
    """
    출력 분포 엔트로피 분석.

    ENGINE_HUB 연결: StatMech_Engine (40_SPATIAL_LAYER)
    자체 구현: Gibbs 엔트로피 -Σ p·ln(p) + 정규화 + 온도 추정
    """

    def __init__(self):
        self._ext_engine = None
        self._try_load_engine()
        self._history: List[float] = []

    def _try_load_engine(self):
        mod = _try_import(
            _ENGINE_HUB / "2_operational" / "40_SPATIAL_LAYER" / "StatMech_Engine",
            "core",
        )
        if mod and hasattr(mod, "StatMechEngine"):
            self._ext_engine = mod.StatMechEngine()

    def analyze_distribution(self, probs: torch.Tensor) -> EntropyResult:
        """
        확률 분포의 엔트로피를 분석한다.

        Args:
            probs: [..., vocab_size] 확률 분포 (softmax 출력)
        """
        p = probs.detach().float().clamp(min=1e-12)
        if p.dim() > 1:
            p = p.reshape(-1, p.shape[-1]).mean(dim=0)

        p = p / p.sum()
        n = p.shape[0]

        entropy = -(p * p.log()).sum().item()
        max_ent = math.log(n)
        norm_ent = entropy / max_ent if max_ent > 0 else 0.0

        temp_est = norm_ent * 2.0 if norm_ent > 0.5 else max(0.1, norm_ent)

        self._history.append(entropy)

        return EntropyResult(
            entropy=entropy,
            max_entropy=max_ent,
            normalized_entropy=norm_ent,
            temperature_estimate=temp_est,
        )

    def analyze_logits(self, logits: torch.Tensor, temperature: float = 1.0) -> EntropyResult:
        """logits → softmax → entropy."""
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        return self.analyze_distribution(probs)

    def entropy_trend(self, window: int = 10) -> dict:
        """최근 엔트로피 추세."""
        h = self._history
        if len(h) < 2:
            return {"trend": "insufficient", "values": h}
        recent = h[-window:] if len(h) >= window else h
        slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        return {
            "trend": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable",
            "slope": slope,
            "mean": sum(recent) / len(recent),
            "latest": recent[-1],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. IntegrationDiagnostic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class IntegrationResult:
    """통합 정보 진단 결과."""
    phi_approx: float = 0.0
    spectral_gap: float = 0.0
    effective_rank: float = 0.0
    verdict: str = ""


class IntegrationDiagnostic:
    """
    내부 상태 통합도(Φ) 진단.

    ENGINE_HUB 연결: IIT_Engine (50_DIAGNOSTIC_LAYER)
    자체 구현: 스펙트럼 분석 기반 Φ 근사
      - spectral gap: 상태 결합 행렬의 제1-제2 고유값 차이
      - effective rank: 고유값 분포의 유효 차원 (Shannon entropy 기반)
    """

    def __init__(self):
        self._ext_engine = None
        self._try_load_engine()

    def _try_load_engine(self):
        mod = _try_import(
            _ENGINE_HUB / "2_operational" / "50_DIAGNOSTIC_LAYER" / "IIT_Engine",
            "core",
        )
        if mod and hasattr(mod, "IITEngine"):
            self._ext_engine = mod.IITEngine()

    def analyze_coupling(self, weight_matrix: torch.Tensor) -> IntegrationResult:
        """
        결합 행렬에서 통합 정보를 근사한다.

        Args:
            weight_matrix: [n, n] 상태 결합 가중치 (dynamics core에서 추출)
        """
        W = weight_matrix.detach().float()

        if self._ext_engine is not None:
            return self._analyze_with_engine(W)
        return self._analyze_builtin(W)

    def _analyze_with_engine(self, W: torch.Tensor) -> IntegrationResult:
        try:
            W_list = W.tolist()
            result = self._ext_engine.analyze(W_list)
            return IntegrationResult(
                phi_approx=result.phi_result.phi if hasattr(result, "phi_result") else 0.0,
                spectral_gap=self._spectral_gap(W),
                effective_rank=self._effective_rank(W),
                verdict=getattr(result, "verdict", ""),
            )
        except Exception:
            return self._analyze_builtin(W)

    def _analyze_builtin(self, W: torch.Tensor) -> IntegrationResult:
        if W.shape[0] < 2 or W.shape[1] < 2:
            return IntegrationResult(verdict="matrix_too_small")

        sg = self._spectral_gap(W)
        er = self._effective_rank(W)

        phi = sg * math.log(max(er, 1.0))

        if phi > 1.0:
            verdict = "highly_integrated"
        elif phi > 0.3:
            verdict = "moderately_integrated"
        elif phi > 0.05:
            verdict = "weakly_integrated"
        else:
            verdict = "fragmented"

        return IntegrationResult(
            phi_approx=phi,
            spectral_gap=sg,
            effective_rank=er,
            verdict=verdict,
        )

    @staticmethod
    def _spectral_gap(W: torch.Tensor) -> float:
        try:
            sym = (W + W.T) / 2.0
            eigvals = torch.linalg.eigvalsh(sym)
            sorted_ev = eigvals.abs().sort(descending=True).values
            if len(sorted_ev) >= 2:
                return (sorted_ev[0] - sorted_ev[1]).item()
            return sorted_ev[0].item() if len(sorted_ev) >= 1 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _effective_rank(W: torch.Tensor) -> float:
        try:
            s = torch.linalg.svdvals(W.float())
            s = s[s > 1e-10]
            if len(s) == 0:
                return 0.0
            p = s / s.sum()
            entropy = -(p * p.log()).sum().item()
            return math.exp(entropy)
        except Exception:
            return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  통합 진단 퍼사드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DynLLMDiagnostics:
    """
    DynLLM 통합 진단 시스템.

    3개 진단 채널을 하나의 인터페이스로 묶는다:
      - convergence: 학습 수렴 진단
      - entropy: 출력 분포 품질
      - integration: 내부 결합도
    """

    def __init__(self, loss_window: int = 50):
        self.convergence = ConvergenceMonitor(window=loss_window)
        self.entropy = EntropyAnalyzer()
        self.integration = IntegrationDiagnostic()

    def record_loss(self, loss: float):
        self.convergence.record(loss)

    def record_logits(self, logits: torch.Tensor, temperature: float = 1.0) -> EntropyResult:
        return self.entropy.analyze_logits(logits, temperature)

    def full_diagnostic(
        self,
        coupling_matrix: Optional[torch.Tensor] = None,
    ) -> dict:
        """전체 진단 리포트."""
        report: dict = {}

        conv = self.convergence.analyze()
        report["convergence"] = {
            "order": conv.convergence_order,
            "lyapunov": conv.lyapunov_estimate,
            "health": conv.health,
            "stable": conv.stable,
            "verdict": conv.verdict,
        }

        report["entropy"] = self.entropy.entropy_trend()

        if coupling_matrix is not None:
            integ = self.integration.analyze_coupling(coupling_matrix)
            report["integration"] = {
                "phi_approx": integ.phi_approx,
                "spectral_gap": integ.spectral_gap,
                "effective_rank": integ.effective_rank,
                "verdict": integ.verdict,
            }

        engines_available = []
        if self.convergence._ext_engine is not None:
            engines_available.append("ConvergenceDynamics")
        if self.entropy._ext_engine is not None:
            engines_available.append("StatMech")
        if self.integration._ext_engine is not None:
            engines_available.append("IIT")
        report["engines_available"] = engines_available

        return report
