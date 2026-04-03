> **한국어 (정본).** English: [README_EN.md](README_EN.md)

# Dynamical LLM Foundation — 개념 정본

**버전**: v0.5.0 (Phase A–E 완료)  
**코드네임**: 대뇌피질형 동역학 언어 엔진  
**저자**: GNJz (Qquarts)  
**위치**: `_staging/Dynamical_LLM_Foundation/`

---

## 이게 뭔가

**Transformer를 복제하지 않는다.**

토큰이 들어올 때마다 내부 상태가 ODE를 따라 흘러가고,
결합하고, 안정화되는 것을 언어 모델의 본체로 삼는 방식이다.

- 기존 LLM: 토큰 → [Attention + FFN] × N층 → 다음 토큰
- 이것: 토큰 → 상태 주입 → ẋ = −λx + F(x, u, m, c; θ) → 상태 판독 → 다음 토큰

**목표**: 누구나 자기만의 언어 지능을 만들어갈 수 있는 동역학 베이스.

---

## 현재 상태 (v0.5.0)

**Phase A–E 전체 완료. 기초 레이어 + 개인화 + 에이전트 통합까지 구축.**

### 모듈 현황
| 모듈 | 파일 | 상태 |
|------|------|------|
| L0 Tokenizer | `tokenizer.py` | char-level, 완료 |
| L1 State Encoder | `state_encoder.py` | nn.Embedding, 완료 |
| L2 Dynamics Core | `dynamics_core.py` | RK4 + VectorField + context coupling + timescale sep, 완료 |
| L3 Memory (4-tier) | `memory.py` | WM(write gate+feedback) + Hebbian(selective) + Episodic, 완료 |
| L4 Readout | `readout.py` | 선형 판독, 완료 |
| L5 Adaptation | `readout.py` | OnlineAdapter + ConsolidationScheduler, 완료 |
| Stability | `stability.py` | TrustGate + RollbackPolicy + SafeStateBuffer + VelocityMonitor, 완료 |
| 통합 모델 | `model.py` | DynLLM 전체 통합, 완료 |
| 학습 | `train.py` | 코퍼스 + 체크포인트, 완료 |
| 추론 | `generate.py` | 인터랙티브, 완료 |
| 평가 | `evaluate.py` | perplexity + diversity + memory utilization, 완료 |
| 개인 메모리 | `personal_memory.py` | SQLite 영구 저장 + crystal + injection, 완료 |
| 증류 브릿지 | `distill_bridge.py` | 대도서관 → DynLLM 자동 증류, 완료 |
| Atom 커넥터 | `Atom/connectors/dynllm_connector.py` | Atom ↔ DynLLM 연결, 완료 |
| 테스트 | `tests/test_dynllm.py` + `test_package_integrity.py` | **65 passed** |

---

## 핵심 수식

```
ẋ = −λx + F(x, u, m, c; θ)

x : 은닉 상태벡터 ∈ ℝ^d
u : 입력 임펄스
m : 메모리 판독 (WM + Hebbian_selective + Episodic)
c : 맥락 결합 (context coupling)
θ : 학습 파라미터
λ : 감쇠 계수 (learnable)

Time-scale:  ẋ_fast = (−λx_fast + F_fast) / τ_fast
             ẋ_slow = (−λx_slow + F_slow) / τ_slow
```

---

## 전체 아키텍처

```
┌────────────────────────────────────────────────────────────┐
│                        Atom Agent                          │
│  dynllm_connector.py — 텍스트 생성 / 메모리 주입 / 증류     │
└───────────┬──────────────────────┬─────────────────────────┘
            │                      │
    ┌───────▼────────┐   ┌────────▼─────────┐
    │ PersonalMemory │   │  DistillBridge   │
    │  (SQLite 영구)  │   │ (대도서관 → 증류)  │
    └───────┬────────┘   └────────┬─────────┘
            │   injection          │   training corpus
    ┌───────▼──────────────────────▼─────────────────────────┐
    │                    DynLLM Model                         │
    │                                                         │
    │  L0: Tokenizer ──────────────────────────────────────   │
    │  L1: State Encoder ──────────────────────────────────   │
    │  L2: Dynamics Core ★                                    │
    │      ├─ VectorField (low-rank coupling + gates)         │
    │      ├─ ContextCoupling (multi-head window)             │
    │      └─ TimescaleSeparator (fast/slow)                  │
    │  L3: Memory (4-tier)                                    │
    │      ├─ Working Memory (PFC + write gate + feedback)    │
    │      ├─ Hebbian (selective recall, top-k sparse)        │
    │      └─ Episodic (sequence trajectory compression)      │
    │  L4: Readout                                            │
    │  L5: OnlineAdapter + ConsolidationScheduler             │
    │  Stability: TrustGate + RollbackPolicy + SafeStateBuf  │
    └─────────────────────────────────────────────────────────┘
```

---

## Phase 로드맵 (완료)

### Phase A: 최소 작동 모델 ✅
- [x] char tokenizer + state encoder + RK4 dynamics + readout
- [x] memory write gate + safe rollback + Hebbian 개별 저장

### Phase B: 학습 가능한 동역학 ✅
- [x] context coupling (multi-head 윈도우)
- [x] time-scale separation (fast/slow, τ learnable)
- [x] refined gating (독립 × 4 + residual)
- [x] VelocityMonitor + spectral norm

### Phase C: 메모리 고도화 ✅
- [x] selective recall (top-k sparse retrieval)
- [x] episodic memory (시퀀스 궤적 압축·회수)
- [x] WM ↔ dynamics core 양방향 (summary feedback)

### Phase D: 온라인 적응 고도화 ✅
- [x] fast weight decay + state adapter
- [x] TrustGate (EMA + warmup)
- [x] ConsolidationScheduler (fast→slow 자동)
- [x] RollbackPolicy (연속 불안정 → 보간 → cooldown)

### Phase E: 개인화 + 에이전트 통합 ✅
- [x] Atom ↔ DynLLM 커넥터 (dynllm_connector.py)
- [x] PersonalMemoryStore (SQLite + knowledge crystals + injection)
- [x] DistillBridge (대도서관 → 증류 버퍼 → 학습 코퍼스)
- [x] 평가 도구 (perplexity, diversity, memory utilization)
- [x] 65 tests passed (`test_dynllm` + `test_package_integrity`)

---

## 개인화 루프

```
사용자 질문 → Atom agent
    ↓
DynLLM.generate() → 응답 시도
    ↓ (품질 부족)
대도서관(Library_of_Babel) → LLM 응답
    ↓
DistillBridge → 증류 버퍼 축적
    ↓
PersonalMemoryStore → 대화 이력 영구 저장
    ↓
MemoryInjector → Hebbian/Episodic에 주입
    ↓
다음 질문에서 DynLLM이 더 개인화된 응답
```

---

## 보수적 경고

1. 기존 Transformer 수준 성능을 바로 기대하면 안 된다
2. 장기 의존성 처리가 생각보다 어렵다
3. 안정한 학습 규칙을 잘못 잡으면 발산한다
4. 온라인 적응은 drift/자기오염 위험 있다
5. 개인 메모리 축적이 충분해야 개인화 효과가 나타난다
6. PyTorch 기반 — stdlib-only 아니다

---

## 인수인계 노트

1. `dynllm/` 패키지: dynamics_core(★), memory, readout, stability, model이 핵심
2. `personal_memory.py`: SQLite 영구 저장 + knowledge crystal + injection
3. `distill_bridge.py`: 대도서관 연결 → 자동 증류 축적
4. `evaluate.py`: perplexity, diversity, memory utilization 계산
5. `Atom/connectors/dynllm_connector.py`: Atom 에이전트가 DynLLM을 쓰는 진입점
6. `train.py` 코퍼스·체크포인트, `generate.py` 추론
7. 향후: 실제 한국어/영어 대규모 코퍼스, subword tokenizer, ATON Nexus 통합
8. 65 tests — 모든 Phase의 forward/backward/generate/memory/adaptation/personal/distill + 패키지 정합성 검증
