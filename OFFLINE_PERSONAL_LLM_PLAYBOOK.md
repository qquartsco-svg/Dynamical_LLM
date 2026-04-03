# Dynamical LLM Foundation — Offline Personal LLM Playbook

## 이 문서의 역할

이 문서는 `외부 LLM 없이 어떻게 운영하는가`에 집중한다.

- 오프라인 본선 루프
- teacher를 언제 붙이고 언제 붙이지 말아야 하는지
- 개인 코어를 작은 데이터로 어떻게 키우는지

사용자 유형별 전체 활용법은 [USER_GUIDE.md](USER_GUIDE.md) 에서 더 넓게 설명한다.

## 목적

이 문서는 `Dynamical_LLM_Foundation`을
외부 LLM API 없이도 개인용 언어 코어로 키워가는 최소 운영 흐름을 정리한다.

핵심 질문은 이것이다.

`인터넷이 끊겨도 내 언어 코어는 계속 자랄 수 있는가?`

이 문서는 그 질문에 대한 보수적인 운영 답안이다.

## 기본 원칙

1. 외부 teacher는 선택 사항이다.
2. 본선은 `개인 코퍼스 + 개인 메모리 + 반복 학습`이다.
3. 처음부터 큰 모델을 목표로 하지 않는다.
4. 작은 코어를 안정적으로 키우는 것이 우선이다.

## 가장 작은 시작

### 1. 코퍼스 준비

아래 중 하나만 있어도 시작할 수 있다.

- 개인 메모 1개 파일
- 프로젝트 문서 묶음
- 대화 로그 일부
- 자주 쓰는 표현과 문장 모음

권장 시작 크기:

- 200~2,000줄
- 너무 크지 않은 텍스트
- 말투와 관심사가 드러나는 자료

### 2. 첫 학습

```bash
cd /Users/jazzin/Desktop/00_BRAIN/_staging/Dynamical_LLM_Foundation
python3 train.py --corpus my_notes.txt --epochs 20 --save out
```

### 3. 첫 생성

```bash
python3 generate.py "오늘 내가 해야 할 일은?" --model out/final_model.pt
```

여기서 보는 것은:

- 내 말투와 비슷한가
- 상태가 쉽게 붕괴하지 않는가
- 프롬프트에 최소한 반응하는가

## 오프라인 성장 루프

```text
개인 문서/메모
-> train.py
-> 첫 DynLLM 코어
-> generate.py
-> personal_memory.py 누적
-> 주기적 재학습
-> 더 개인화된 DynLLM
```

이 루프가 오프라인 본선이다.

## 메모리를 언제 붙여야 하나

추천 순서는 이렇다.

1. 먼저 작은 학습이 돌아가는지 확인
2. 생성이 아주 기본 수준으로 되는지 확인
3. 그다음 `personal_memory.py`를 붙여 사용자 이력 축적

즉 메모리는 첫 번째가 아니라 두 번째 단계다.

## 외부 teacher는 언제 붙이나

외부 teacher는 아래 조건일 때만 붙이는 것이 좋다.

- 기본 생성이 이미 동작한다
- 내 코퍼스만으로는 부족한 응답 품질을 보완하고 싶다
- teacher 응답을 필터링해서 증류할 운영 규칙이 있다

즉 `distill_bridge.py`는 가속기이지, 필수 부품은 아니다.

## 개인 사용자 운영 로드맵

### Step 1
- 작은 코퍼스로 첫 모델 생성

### Step 2
- 생성 품질과 붕괴 여부 관찰

### Step 3
- 메모리 저장과 회상 주입 추가

### Step 4
- 주기적 재학습

### Step 5
- 필요하면 teacher 증류 추가

### Step 6
- 마지막에 Atom 같은 상위 에이전트와 연결

## 제품 개발자 운영 로드맵

### Phase A
- foundation 자체가 안정적으로 학습/생성되는지 확인

### Phase B
- memory-safe 운영 규칙 추가

### Phase C
- connector 추가

### Phase D
- telemetry / rollback / config schema 강화

### Phase E
- 사용자 UI나 에이전트 제품과 연결

## 무엇을 조심해야 하나

1. char-level은 느리고 품질 한계가 있다
2. 온라인 적응은 drift 위험이 있다
3. 작은 코퍼스는 쉽게 과적합될 수 있다
4. 외부 teacher를 붙이면 오히려 개인 코어 정체성이 흐려질 수 있다

## 한 줄 정리

**이 엔진의 본선은 “외부 LLM이 없어도 개인이 자기 언어 코어를 계속 키워갈 수 있게 하는 것”이다.**
