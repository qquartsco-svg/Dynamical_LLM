> **한국어 (정본).** English: [README_EN.md](README_EN.md)

# Dynamical LLM Foundation

**버전:** `v0.5.0` — Phase A–F 기반 base implementation

문서:
- [사용설명서](USER_GUIDE.md)
- [로드맵](ROADMAP.md)
- [오프라인 플레이북](OFFLINE_PERSONAL_LLM_PLAYBOOK.md)
- [시스템 연결도](SYSTEM_CONNECTION_MAP.md)
- [실험 로그](EXPERIMENT_LOG.md)

Transformer 구조를 복제하지 않는다.  
토큰이 들어올 때마다 내부 상태가 ODE를 따라 진화하고, 4단 메모리와 결합하며, 온라인으로 적응하는 **동역학 언어 모델의 기초 레이어**다.

핵심 철학은 `외부 LLM이 끊겨도 살아남는 개인 대뇌피질 코어`다.
대형 LLM API는 teacher나 보조 자료원으로 붙을 수 있지만, 이 엔진의 존재 이유는 외부 서비스가 없어도 개인이 자기 언어 코어를 계속 키워갈 수 있게 하는 데 있다.

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
L0  Token Interface      문자/바이트 → 정수 ID (char 또는 byte 선택)
L1  State Encoder        ID → 초기 상태벡터 x₀
L2  Dynamics Core        ODE 진화: context coupling + time-scale separation + refined gating
L3  Memory               단기(hidden x) · 작업기억(PFC) · 헤비안(장기 연상) · 에피소딕
L4  Readout              상태 → vocabulary logits
L5  Online Adaptation    fast/slow weights + trust gate + consolidation
```

---

## 구현 상태

| Phase | 내용 | 핵심 결과물 |
|-------|------|------------|
| A | 최소 동작 모델 | `train.py`, `generate.py`, SafeStateBuffer, 헤비안 단일 저장 |
| B | 학습 가능 동역학 | ContextCoupling, TimescaleSeparator, VelocityMonitor, 스펙트럴 정규화 |
| C | 메모리 강화 | SelectiveRecall, EpisodicMemory, WM ↔ Dynamics 양방향 피드백 |
| D | 온라인 적응 강화 | FastWeightDecay, StateAdapter, ConsolidationScheduler, RollbackPolicy |
| E | 개인화 + 외부 브리지 준비 | PersonalMemoryStore, MemoryInjector, DistillBridge |
| F | 시스템 거버넌스 연결 | `system_bridge.py`, Atom/Athena/Aton/Pharaoh 경계 계약 |

위 표는 `구조적 base implementation` 기준이다.  
즉 레이어와 경계 계약은 들어왔지만, 실사용 성숙도는 tokenizer 품질, 학습 기록, connector, benchmark 쪽이 아직 더 올라와야 한다.

---

## 파일 구조

```
Dynamical_LLM_Foundation/
├── dynllm/
│   ├── __init__.py
│   ├── tokenizer.py          # L0 — DynTokenizer (char) + ByteTokenizer (byte)
│   ├── state_encoder.py      # L1
│   ├── dynamics_core.py      # L2 — ContextCoupling, TimescaleSeparator
│   ├── memory.py             # L3 — WorkingMem, HebbianMem, EpisodicMem
│   ├── readout.py            # L4+L5 — OnlineAdapter, ConsolidationScheduler
│   ├── stability.py          # TrustGate, VelocityMonitor, RollbackPolicy
│   ├── model.py              # 전체 통합 DynLLM
│   ├── personal_memory.py    # Phase E — PersonalMemoryStore, MemoryInjector
│   ├── distill_bridge.py     # Phase E — DistillBuffer, DistillBridge (optional teacher path)
│   ├── evaluate.py           # Phase E — perplexity, diversity, memory util
│   └── system_bridge.py      # Phase F — DynLLM ↔ Atom/Athena/Aton/Pharaoh 거버넌스 계약
├── train.py
├── generate.py
├── examples/
│   └── run_dlm.py
├── tests/
│   ├── test_dynllm.py
│   ├── test_smoke_entrypoints.py
│   ├── test_system_bridge.py
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

# 최소 end-to-end 예제
python3 examples/run_dlm.py
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

## 오프라인 개인 LLM 전략

이 엔진은 두 모드로 읽는 것이 맞다.

- `teacher-assisted`
  - 외부 LLM이나 내부 대도서관 응답을 distillation 재료로 참고
- `offline-first`
  - 외부 LLM 없이 개인 코퍼스, 개인 메모리, 장기 누적 학습만으로 성장

중요한 점:

- `distill_bridge.py`는 선택 사항이다
- 개인 사용의 본선은 `train.py + personal_memory.py + generate.py`
- 외부 teacher가 없어도 이 엔진은 성립해야 한다
- 상용화 가치도 “API 래퍼”가 아니라 “개인 전용 언어 코어” 쪽에 있다

---

## 외부 연결 준비

Phase E 기준으로 `personal_memory.py`, `distill_bridge.py`가 들어와 있어서,
개인 메모리와 optional teacher 응답을 DynLLM 쪽으로 연결할 준비가 되어 있다.
다만 이 저장소는 `외부 LLM 연결`이 아니라 `개인 코어 형성`이 본선이고,
`Atom.connectors.dynllm_connector` 자체는 이 저장소 안에 포함돼 있지 않다.

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
- memory recall rate
- system bridge / governance contracts
- package integrity

검증 수치는 환경에 따라 다르게 읽어야 한다.

| 구분 | 명령 | 의미 | 현재 결과 |
|------|------|------|-----------|
| torch full suite | `python3 scripts/release_check.py` 내부 pytest | torch 런타임에서의 전체 suite | **81 passed** |
| release gate | `python3 scripts/release_check.py` | 패키지 정합성 + 서명 + 테스트 통합 체크 | `OK` |
| 서명 검증 | `python3 scripts/verify_signature.py` | 39개 파일 해시 일치 여부 | `passed=39 failed=0 missing=0` |

`torch`가 없는 환경에서는 runtime 테스트가 보수적으로 skip될 수 있다.

---

## 한계

- Transformer급 사전학습 성능을 목표로 하지 않는다
- 지금은 `개인 오프라인 코어`로서의 가치를 우선 본다
- 장기 drift 통제는 TrustGate + RollbackPolicy로 부분 대응 중
- 대규모 코퍼스 학습은 아직 검증 전
- byte-level 한국어는 더 많은 epoch이 필요 (UTF-8 3바이트 조합 학습)

---

## 토크나이저

두 가지 모드를 지원한다.

| 모드 | vocab | 한국어 | 특징 |
|------|-------|--------|------|
| `DynTokenizer` (char) | 코퍼스 의존 | fit 필요 | 단순, 투명 |
| `ByteTokenizer` (byte) | 고정 260 | 즉시 대응 | 모든 언어, 재학습 불필요 |

```bash
# byte-level 학습
python3 train.py --byte --epochs 20

# byte-level 생성
python3 generate.py --byte "안녕하세요"
```

---

## 실험 결과 요약

[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 에 상세 기록. 아래는 핵심만.

| 실험 | tokenizer | corpus | loss drop | final ppl | time (CPU) |
|------|-----------|--------|-----------|-----------|------------|
| 1 | char(22) | EN 630자 | 0.54 | 9.1 | 90s |
| 2 | byte(260) | EN 630자 | 1.26 | 7.5 | 77s |
| 3 | byte(260) | KR 1845바이트 | 2.23 | 2.5 | 239s |

핵심 관찰:
- ODE dynamics core가 실제로 학습한다 — 모든 실험에서 loss 일관 하락
- 감쇠(λ)가 자동 적응 — 한국어에서 0.44까지 상승 (모델이 안정점을 스스로 찾음)
- ByteTokenizer가 char보다 convergence 2.3배 빠름
- 추론 속도: **~1,170 tok/s** (CPU, 메모리 OFF) / **~950 tok/s** (메모리 ON)

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
