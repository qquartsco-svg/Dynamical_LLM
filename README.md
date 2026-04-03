> **한국어 (정본).** English: [README_EN.md](README_EN.md)

# Dynamical LLM Foundation

**버전:** `v0.5.0` — Phase A–E 기반 구현

Transformer 구조를 복제하지 않는다.  
토큰이 들어올 때마다 내부 상태가 ODE를 따라 진화하고, 4단 메모리와 결합하며, 온라인으로 적응하는 **동역학 언어 모델의 기초 레이어**다.

---

## 핵심 수식

```text
ẋ = -λx + F(x, u, m; θ)
```

| 기호 | 의미 |
|------|------|
| `x` | 은닉 상태벡터 |
| `λ` | 감쇠 계수 (학습 가능, 발산 방지) |
| `u` | 현재 토큰 입력 임펄스 |
| `m` | working/hebbian memory 신호 |
| `θ` | 벡터장 파라미터 (학습 가능) |

---

## 아키텍처 — 6개 레이어

```
L0  Token Interface      문자 → 정수 ID
L1  State Encoder        ID → 초기 상태벡터 x₀
L2  Dynamics Core        ODE 진화: context coupling + time-scale separation + refined gating
L3  Memory               단기(hidden x) · 작업기억(PFC) · 헤비안(장기 연상) · 에피소딕
L4  Readout              상태 → vocabulary logits
L5  Online Adaptation    fast/slow weights + trust gate + consolidation
```

---

## 완료된 Phase

| Phase | 내용 | 핵심 결과물 |
|-------|------|------------|
| A | 최소 동작 모델 | `train.py`, `generate.py`, SafeStateBuffer, 헤비안 단일 저장 |
| B | 학습 가능 동역학 | ContextCoupling, TimescaleSeparator, VelocityMonitor, 스펙트럴 정규화 |
| C | 메모리 강화 | SelectiveRecall, EpisodicMemory, WM ↔ Dynamics 양방향 피드백 |
| D | 온라인 적응 강화 | FastWeightDecay, StateAdapter, ConsolidationScheduler, RollbackPolicy |
| E | 개인화 + 외부 브리지 준비 | PersonalMemoryStore, MemoryInjector, DistillBridge |

---

## 파일 구조

```
Dynamical_LLM_Foundation/
├── dynllm/
│   ├── __init__.py
│   ├── tokenizer.py          # L0
│   ├── state_encoder.py      # L1
│   ├── dynamics_core.py      # L2 — ContextCoupling, TimescaleSeparator
│   ├── memory.py             # L3 — WorkingMem, HebbianMem, EpisodicMem
│   ├── readout.py            # L4+L5 — OnlineAdapter, ConsolidationScheduler
│   ├── stability.py          # TrustGate, VelocityMonitor, RollbackPolicy
│   ├── model.py              # 전체 통합 DynLLM
│   ├── personal_memory.py    # Phase E — PersonalMemoryStore, MemoryInjector
│   ├── distill_bridge.py     # Phase E — DistillBuffer, DistillBridge
│   └── evaluate.py           # Phase E — perplexity, diversity, memory util
├── train.py
├── generate.py
├── tests/
│   ├── test_dynllm.py        # 63개 단위/통합 테스트
│   └── test_package_integrity.py
└── scripts/
    ├── generate_signature.py
    ├── verify_signature.py
    ├── verify_package_identity.py
    ├── release_check.py
    └── cleanup_generated.py
```

---

## 빠른 시작

```bash
pip install -e .
```

```bash
# 학습 (기본)
python3 train.py --epochs 20

# 코퍼스 파일로 학습
python3 train.py --corpus mytext.txt --epochs 50 --save ckpt.pt

# 생성
python3 generate.py "안녕하세요"

# 체크포인트에서 생성
python3 generate.py "hello" --model ckpt.pt --max_tokens 200
```

```python
from dynllm.model import DynLLM, DynLLMConfig
import torch

cfg = DynLLMConfig(
    vocab_size=128,
    d_state=128,
    use_memory=True,
    context_window=8,
    use_timescale_sep=True,
)
model = DynLLM(cfg)
ids = torch.randint(1, 128, (2, 32))
logits = model(ids)
print(logits.shape)          # [2, 32, 128]
print(model.count_parameters())
```

---

## 상용화 가치

이 엔진의 상용화 가치는 “범용 대형 LLM을 바로 대체한다”는 쪽보다 아래에 더 가깝다.

- on-device 개인화 언어 엔진 베이스
- 적응형 에이전트 코어
- 메모리/적응/드리프트 통제가 필요한 사설 LLM 런타임
- 연구용 동역학 구조를 제품형 코어로 옮기는 브리지

즉 현재 강점은 `성능 경쟁`보다 `개인화`, `구조 해석 가능성`, `적응성`, `상태 기반 제어성`이다.

---

## 외부 연결 준비

Phase E 기준으로 `personal_memory.py`, `distill_bridge.py`가 들어와 있어서,
개인 메모리와 외부 teacher 응답을 DynLLM 쪽으로 연결할 준비가 되어 있다.
다만 `Atom.connectors.dynllm_connector` 자체는 이 저장소 안에 포함돼 있지 않다.

```python
from dynllm.personal_memory import PersonalMemoryStore
from dynllm.distill_bridge import DistillBuffer, DistillSample

store = PersonalMemoryStore(":memory:")
buffer = DistillBuffer()

store.log_interaction("오늘 날씨가 좋네", "응, 오늘은 맑은 편이야.")
buffer.add(DistillSample("오늘 날씨가 좋네", "응, 오늘은 맑은 편이야."))
```

---

## 검증

```bash
python3 -m pytest tests/ -q
# torch 환경: 전체 runtime 테스트 수행
# no-torch 환경: torch 의존 테스트는 skip될 수 있음

python3 scripts/release_check.py
# verify_package_identity + verify_signature + pytest
```

테스트 범주:

- tokenizer / state encoder / integrator
- dynamics core (VectorField, ContextCoupling, TimescaleSeparator)
- memory (Working, Hebbian, Episodic, MemorySystem)
- stability (TrustGate, VelocityMonitor, RollbackPolicy)
- readout (OnlineAdapter, ConsolidationScheduler)
- full model forward / backward / generate
- train / generate smoke entrypoints
- personal memory / distill bridge / evaluate
- package integrity

---

## 한계

- Transformer급 사전학습 성능을 목표로 하지 않는다
- 현재 char-level tokenizer — subword 확장은 Phase F 이후
- 장기 drift 통제는 TrustGate + RollbackPolicy로 부분 대응 중
- 대규모 코퍼스 학습은 아직 검증 전

---

## 무결성 서명

```bash
python3 scripts/generate_signature.py   # SIGNATURE.sha256 재생성
python3 scripts/verify_signature.py     # 검증
```

상세: [BLOCKCHAIN_INFO.md](BLOCKCHAIN_INFO.md)

---

## 이 패키지가 하지 않는 일

- Transformer를 그대로 대체한다고 주장하지 않는다
- 대규모 LLM과 같은 성능을 약속하지 않는다
- "정답"을 내리는 엔진이 아니라, 동역학 언어 모델의 **기초 구조**를 세우는 foundation이다
- 현재 포지셔닝은 완성형 상용 LLM보다, 개인화 가능한 동역학 언어 엔진 베이스에 가깝다
- 현재 포지셔닝은 완성형 상용 LLM보다, 개인화 가능한 동역학 언어 엔진 베이스에 가깝다
