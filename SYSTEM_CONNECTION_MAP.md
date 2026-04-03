# Dynamical LLM Foundation — System Connection Map

## 핵심 해석

`DynLLM`은 전체 시스템 그 자체가 아니다.

이 엔진은:

- `Atom`의 실행체 전체도 아니고
- `ATHENA`의 판단체 전체도 아니고
- `ATON`의 넥서스 전체도 아니다

그 대신 **대뇌피질형 언어-상태 코어**다.

즉:

- `Atom` = 개인용 창의·창발 실행체
- `ATHENA` = 군주형 공공 판단·권고 체계
- `DynLLM` = 그 사이에서 생각/언어/기억을 생성하는 cortical core
- `ATON` = 전체 시스템 넥서스/오케스트레이터
- `PHARAOH` = 의견 수렴·공공 합의층
- `USER` = 최종 결정권자

그리고 중요한 원칙 하나가 더 있다.

- `DynLLM`은 외부 LLM API가 끊겨도 살아남는 개인 대뇌피질 코어여야 한다
- 외부 teacher나 대도서관 연결은 보조 경로이지 본체가 아니다

## 원하는 운영 흐름

```text
사용자
-> Atom / Athena
-> Aton
-> 전체 시스템 orchestration
-> Athena recommendation
-> Pharaoh consensus
-> 사용자 최종 결정
```

## 두 가지 운용 모드

### 1. 독립형

시스템이 넥서스에 연결되지 않았을 때:

```text
사용자
-> Atom (개인형 독립 지능)
-> DynLLM cortical draft
-> Athena review
-> 사용자 결정
```

이 모드에서:

- `Atom`은 개인 창의/실행체
- `Athena`는 개인에게 권고하는 감독층
- `DynLLM`은 개인 언어/생각 코어

즉 독립 지능체처럼 작동할 수 있다.

### 2. 넥서스 연결형

시스템이 `ATON`에 연결되었을 때:

```text
사용자
-> Atom / Athena local layer
-> DynLLM cortical draft
-> ATON nexus orchestration
-> ATHENA recommendation
-> PHARAOH opinion collection
-> 사용자 최종 결정
```

이 모드에서:

- `Atom/Athena`의 출력은 로컬 종료가 아니다
- `ATON`이 전체 운영 맥락으로 끌어올린다
- 필요 시 `PHARAOH`로 의견 수렴이 간다
- 최종 결정은 여전히 사용자에게 남는다

## DynLLM의 자리

DynLLM은 이 구조에서 다음 역할을 맡는다.

1. 입력을 받아 상태를 진화시킴
2. 기억과 결합해 후보 텍스트/의도를 생성
3. 그 초안을 `ATHENA`가 검토할 수 있게 메타데이터와 함께 넘김

즉 `DynLLM -> ATHENA` 경계에서 중요한 것은:

- draft text
- confidence
- state summary
- memory sources
- risk tags
- action hints

그리고 `DynLLM -> Atom` 경계에서 중요한 것은:

- 개인 메모리 주입
- 로컬 세션 상태
- 사용자 말투/선호 축적
- 외부 teacher 없이도 유지되는 자가 성장 루프

이 저장소의 [dynllm/system_bridge.py](dynllm/system_bridge.py)는
그 계약을 최소 형태로 고정한다.

## Bridge Payload

`system_bridge.py` 기준 최소 계약은 아래처럼 읽으면 된다.

### DynLLMDraft

| 필드 | 필수 | 의미 |
|------|------|------|
| `text` | 예 | DynLLM이 만든 cortical draft |
| `confidence_0_1` | 아니오 | 초안에 대한 내부 신뢰도 |
| `state_summary` | 아니오 | 상태 요약 문자열 |
| `memory_sources` | 아니오 | 어떤 기억 원천이 개입했는지 |
| `risk_tags` | 아니오 | drift, public-risk 등 위험 태그 |
| `action_hints` | 아니오 | 상위 계층에 주는 실행 힌트 |

### AthenaRecommendation

| 필드 | 필수 | 의미 |
|------|------|------|
| `stage` | 예 | `positive / neutral / cautious / negative` |
| `advisory` | 예 | ATHENA 권고 문장 |
| `allow_execute` | 예 | 실행 허용 여부 |
| `requires_public_consensus` | 아니오 | Pharaoh 합의층 필요 여부 |
| `evidence_tags` | 아니오 | 판단 근거 태그 |

### GovernancePacket

| 필드 | 필수 | 의미 |
|------|------|------|
| `mode` | 예 | `standalone_personal` 또는 `nexus_connected` |
| `draft` | 예 | DynLLM 초안 |
| `athena` | 예 | Athena 판단 |
| `route` | 예 | 최종 라우팅 경로 |
| `pharaoh_opinions` | 아니오 | 수렴된 의견 목록 |
| `final_user_required` | 예 | 최종 사용자 판단 필요 여부 |

## Route Rules

### Standalone

- `negative` -> `hold_and_revise`
- `cautious` -> `athena_to_user`
- 그 외 -> `direct_to_user`

### Nexus-connected

- `negative` -> `hold_and_revise`
- `cautious` 또는 `requires_public_consensus=True` -> `aton_to_pharaoh_to_user`
- 그 외 -> `athena_to_user`

## 왜 이렇게 쌓아야 하나

이유는 두 가지다.

1. 개인 지능과 공공 거버넌스를 같은 층에 섞지 않기 위해
2. 사용자 최종 결정을 항상 남겨두기 위해
3. 개인 코어와 공공 거버넌스 코어를 서로 오염시키지 않기 위해

즉:

- DynLLM이 직접 통치하지 않는다
- Athena가 직접 최종 결정을 강제하지 않는다
- Pharaoh는 의견 수렴을 한다
- 최종 책임은 사용자에게 남는다

## 연결 원칙

### Atom
- 창의/창발/개인적 흐름
- 실행 제안의 1차 주체

### DynLLM
- 언어 상태 진화
- 기억 결합
- cortical draft 생성

### ATHENA
- 위험, 공공성, 거시적 판단, 보수성
- `positive / neutral / cautious / negative`

### ATON
- 각 엔진/에이전트의 상위 오케스트레이션
- 어떤 흐름을 Pharaoh까지 올릴지 결정

### PHARAOH
- 다수 의견 / 공공 관점 / 합의 흐름

### USER
- 최종 승인/거부/보류

## 한 줄 정리

**DynLLM은 Atom과 Athena 사이, 그리고 Aton으로 들어가기 전의 대뇌피질형 언어-상태 코어로 연결되어야 한다.**
