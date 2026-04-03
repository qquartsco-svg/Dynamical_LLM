# Dynamical LLM Foundation — 로드맵

## 목표

`Dynamical_LLM_Foundation`을  
**연구용 동역학 언어 모델 MVP -> 개인화 가능한 엔진 베이스 -> 에이전트 코어 런타임**  
으로 키우는 로드맵이다.

## 현재 위치

현재는 `v0.5.0`, Phase A–F 기반 구현이다.

이미 있는 것:

- tokenizer
- state encoder
- dynamics core
- 4-tier memory
- readout
- online adaptation skeleton
- personal memory
- distillation bridge
- evaluation helper
- system governance bridge

아직 약한 것:

- tokenizer 품질
- 실제 학습 품질 기록
- actual Atom connector
- long-context benchmark
- runtime packaging for real end users
- offline-first workflow examples

## 단계별 로드맵

### Phase G — 사용성 강화

목표:

- 실제 사용자가 “한 번 학습하고 한 번 생성해보는” 수준까지 쉽게 만들기

필수 작업:

- byte/subword tokenizer 추가
- training preset 추가
- sample corpus / sample config 추가
- train/generate 결과물 저장 규칙 정리
- README에 실제 예시 출력 추가

성공 기준:

- 새 사용자가 문서만 보고 학습/생성까지 따라할 수 있음

### Phase H — 개인 메모리 강화

목표:

- 개인 LLM 베이스로서 의미를 갖도록 메모리 축을 강화

필수 작업:

- PersonalMemoryStore 자동 적재기
- 중요도/태그/회상 빈도 정책
- memory injection scheduling
- drift-safe recall 정책

성공 기준:

- 대화/메모가 누적되고, 생성 결과에 개인 정보 흐름이 드러남

### Phase I — Distillation Loop

목표:

- 큰 LLM을 teacher로 사용해 DynLLM을 점진적으로 키우는 루프 구축

필수 작업:

- DistillBuffer 품질 필터링
- confidence / source weighting
- small-batch distillation training loop
- teacher/student 비교 평가

성공 기준:

- teacher 응답을 축적하며 student DynLLM이 점진적으로 개선됨

### Phase J — Agent Connector

목표:

- Atom 또는 다른 앱과 실제 연결

필수 작업:

- 별도 connector 저장소 또는 `connectors/` 계층
- prompt/event -> memory injection bridge
- response logging
- session state handoff

성공 기준:

- 외부 앱이 DynLLM을 실제 코어로 호출할 수 있음

### Phase K — Runtime Hardening

목표:

- 상용화 가능한 품질 기준으로 runtime을 단단하게 만들기

필수 작업:

- checkpoint/version policy
- reproducible evaluation suite
- failure/rollback telemetry
- config schema
- packaging / distribution flow

성공 기준:

- 테스트/서명/평가/버전 흐름이 안정적으로 반복 가능함

## 사용자별 로드맵

### 개인 사용자

1. 작은 코퍼스 학습
2. 생성 확인
3. personal memory 연결
4. 주기적 재학습 또는 증류 연결

### 연구자

1. dynamics core 검증
2. memory 구조 비교
3. adaptation 정책 비교
4. tokenizer/benchmark 확장

### 제품 개발자

1. minimal runtime 확보
2. connector 추가
3. memory-safe policy 추가
4. telemetry / evaluation / packaging 추가

## 무엇을 먼저 해야 하는가

가장 좋은 다음 순서는 이렇다.

1. tokenizer 업그레이드
2. 실제 train/generate 샘플 기록
3. memory injection workflow
4. distillation workflow
5. app/agent connector
6. offline-only usage examples

## 한 줄 정리

**이 엔진의 확장은 “더 큰 모델”보다 “더 개인화되고, 더 안정적이고, 더 연결 가능하며, 외부 LLM 없이도 살아남는 동역학 코어” 쪽으로 가야 한다.**
