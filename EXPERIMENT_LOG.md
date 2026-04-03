# Dynamical LLM Foundation — Experiment Log

## 목적

이 문서는 `Dynamical_LLM_Foundation`의 실제 학습/생성 실험 기록을 남기기 위한 자리다.

지금 단계에서 이 문서가 중요한 이유는:

- foundation이 실제로 어떤 코퍼스에서 어떻게 거동하는지 남기기 위해
- README의 개념 설명을 실제 실험 근거와 연결하기 위해
- drift, 안정성, 품질 변화를 추적하기 위해

## 기록 규칙

각 실험은 아래 항목을 남기는 것을 권장한다.

| 항목 | 설명 |
|------|------|
| date | 실험 날짜 |
| corpus | 사용한 코퍼스 설명 |
| tokenizer | char / byte / subword 등 |
| config | 주요 DynLLMConfig |
| epochs | 학습 epoch |
| optimizer / lr | 사용한 최적화 설정 |
| loss/perplexity | 가능한 경우 정량 지표 |
| sample prompt | 테스트 프롬프트 |
| sample output | 샘플 출력 |
| drift / collapse | 발산, 반복, 붕괴 여부 |
| note | 관찰 메모 |

## Current Baseline

- torch full suite: **81 passed** (ByteTokenizer 9개 추가)
- `release_check.py`: `OK`
- 서명: `passed=39 failed=0 missing=0`

---

## Experiment 0 — Smoke baseline

- date: 2026-04-04
- corpus: built-in minimal text / package smoke baseline
- tokenizer: char-level
- config: base implementation sanity check
- epochs: n/a
- optimizer / lr: n/a
- loss/perplexity: not recorded
- sample prompt: see smoke tests
- sample output: runtime-specific
- drift / collapse: not observed
- note: local smoke `6 passed, 2 skipped` (no torch), full suite `71→81 passed`

---

### Experiment 1 — Char-level, 영어 내장 코퍼스

- date: 2026-04-04
- corpus: `"the brain is a dynamical system..." × 10` (약 630자)
- tokenizer: **char** (vocab=22)
- config: `d_state=32, seq_len=16, batch=4, context_window=4, timescale=off`
- epochs: 5
- optimizer / lr: AdamW / 1e-3, cosine schedule
- loss/perplexity: `2.7475 → 2.2040` (ppl 15.6 → 9.1)
- sample prompt: `"the b"`
- sample output: `"raiiy pl  a iiiyinidr<UNK>ap"`
- drift / collapse: 없음. λ = 0.107 → 0.136 (감쇠 자연 증가)
- note: 5 epoch에서도 loss가 뚜렷하게 하락. 아직 의미 있는 텍스트는 아님. **34,680 params, 90초 (CPU)**

---

### Experiment 2 — Byte-level, 영어 내장 코퍼스

- date: 2026-04-04
- corpus: 실험 1과 동일
- tokenizer: **byte** (vocab=260, fixed)
- config: 실험 1과 동일
- epochs: 5
- optimizer / lr: AdamW / 1e-3, cosine schedule
- loss/perplexity: `3.2753 → 2.0119` (ppl 26.5 → 7.5)
- sample prompt: `"the b"`
- sample output: `"mmib iin aaaaaiapdyyanbmocin n sn athwme"`
- drift / collapse: 없음. λ = 0.113 → 0.176 (char보다 큰 감쇠 학습)
- note: **초기 loss가 높지만 drop이 1.26으로 char(0.54)보다 2.3배 크다**. Vocab 고정 260 덕에 한국어 확장 시 재학습 불필요. **50,150 params, 77초 (CPU)** — char보다 14% 빠름

---

### Experiment 3 — Byte-level, 한국어 코퍼스

- date: 2026-04-04
- corpus: `"뇌는 동역학 시스템이다. 기억은 패턴 완성이다..." × 15` (약 1,845 UTF-8 바이트)
- tokenizer: **byte** (vocab=260, fixed)
- config: 실험 1과 동일
- epochs: 5
- optimizer / lr: AdamW / 1e-3, cosine schedule
- loss/perplexity: `3.1337 → 0.9037` (ppl 23.0 → 2.5)
- sample prompt: `"뇌는"`
- sample output: garbled UTF-8 (바이트 수준에서 아직 유효한 3바이트 시퀀스를 학습하지 못함)
- drift / collapse: 없음. λ = 0.162 → 0.440 (한국어 3바이트 구조에 적응하며 감쇠 크게 상승)
- note: **loss drop 2.23으로 가장 크고 ppl 2.5로 수렴**. UTF-8 바이트 조합이 맞지 않아 디코딩 깨짐. 한국어 byte-level은 더 많은 epoch 필요. **50,150 params, 239초 (CPU)** — 코퍼스가 3배 커서 시간 비례 증가

---

## 핵심 관찰

1. **ODE dynamics core가 실제로 학습한다** — loss가 모든 실험에서 일관되게 하락
2. **감쇠(λ)가 자동으로 적응한다** — 한국어 byte-level에서 λ가 0.44까지 올라감. 모델이 스스로 안정점을 찾는 증거
3. **ByteTokenizer가 char보다 convergence 속도 우위** — loss drop 2.3배, 시간 14% 단축
4. **한국어 byte-level은 epoch 부족** — UTF-8 3바이트 조합 학습에 더 많은 반복 필요
5. **발산/붕괴 없음** — TrustGate/RollbackPolicy 개입 사례 0건

---

## 비교 요약

| 실험 | tokenizer | corpus | loss drop | final ppl | time | params |
|------|-----------|--------|-----------|-----------|------|--------|
| 1 | char(22) | EN 630자 | 0.54 | 9.1 | 90s | 34,680 |
| 2 | byte(260) | EN 630자 | 1.26 | 7.5 | 77s | 50,150 |
| 3 | byte(260) | KR 1845바이트 | 2.23 | 2.5 | 239s | 50,150 |

---

## Inference Speed Benchmark (2026-04-04, CPU)

환경: MacBook Air, Python 3.13, torch CPU, `n_substeps=2`

| Config | Params | tok/s | 50 tokens (ms) |
|--------|--------|-------|-----------------|
| small/byte (d=32) | 36,677 | 1,178 | 42 |
| medium/byte (d=64) | 107,909 | 1,171 | 43 |
| medium/byte+mem (d=64) | 161,478 | 961 | 52 |
| medium/byte+timescale (d=64) | 112,071 | 944 | 53 |

관찰:
- **메모리 OFF 기준 ~1,170 tok/s** (CPU, 랜덤 초기화)
- 메모리/타임스케일 켜면 ~18% 하락 (952 tok/s) — 허용 범위
- d_state 32 → 64 올려도 속도 차이 거의 없음 (병목이 ODE substep)
- MPS (Apple Silicon GPU) 시 추가 가속 기대

---

## Entry Template

```text
### Experiment N
- date:
- corpus:
- tokenizer:
- config:
- epochs:
- optimizer / lr:
- loss/perplexity:
- sample prompt:
- sample output:
- drift / collapse:
- note:
```

## 한 줄 정리

**이 문서는 Dynamical LLM Foundation이 "개념적으로 가능하다"를 넘어 "실제로 어떤 조건에서 어떻게 반응하는가"를 누적 기록하는 운영 로그다.**
