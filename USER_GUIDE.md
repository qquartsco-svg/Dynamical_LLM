# Dynamical LLM Foundation — 사용설명서

## 이 문서의 역할

이 문서는 `누가 어떻게 사용하는가`에 집중한다.

- 사용자 유형별 추천 사용 순서
- 실제 첫 학습/첫 생성 흐름
- 개인 사용자 / 연구자 / 제품 개발자 관점의 차이

`오프라인 운영 루프` 자체는 [OFFLINE_PERSONAL_LLM_PLAYBOOK.md](OFFLINE_PERSONAL_LLM_PLAYBOOK.md) 에서 더 직접적으로 다룬다.

## 누구를 위한 엔진인가

이 엔진은 아래 사용자에게 맞는다.

- 자기만의 개인 LLM 코어를 직접 키우고 싶은 사람
- Transformer 외 구조를 실험하고 싶은 연구자/개발자
- 메모리, 적응, drift 제어가 중요한 private runtime을 만들고 싶은 사람
- 에이전트의 내부 언어 코어를 더 해석 가능한 상태 기반 구조로 만들고 싶은 사람

이 엔진은 아래 목적에는 아직 맞지 않는다.

- 바로 범용 상용 챗봇을 대체하려는 목적
- 대규모 사전학습 LLM과 같은 성능 기대
- 즉시 production-scale inference serving

## 이 엔진을 어떻게 봐야 하는가

이 저장소는 “완성형 LLM 앱”이 아니라 **개인화 가능한 동역학 언어 엔진 베이스**다.

그리고 더 정확히 말하면:

- 외부 LLM API가 있으면 teacher로 참고할 수 있고
- 외부 LLM API가 끊겨도 개인 코퍼스와 개인 메모리만으로 계속 자랄 수 있어야 한다

즉 이 엔진의 본선은 `offline-first personal core`다.

가장 좋은 사용 순서는 이렇다.

1. 작은 개인 코퍼스로 기본 상태를 만든다
2. 생성 결과를 본다
3. 개인 메모리를 붙인다
4. teacher 응답 증류를 붙인다
5. 마지막에 외부 에이전트/앱 커넥터를 붙인다

## 빠른 사용 순서

### 1. 설치

```bash
cd /Users/jazzin/Desktop/00_BRAIN/_staging/Dynamical_LLM_Foundation
pip install -e .
```

### 2. 가장 작은 학습

```bash
python3 train.py --epochs 20
```

이건 내장 코퍼스로 작은 char-level 학습을 돌린다.

### 3. 내 텍스트로 학습

```bash
python3 train.py --corpus my_notes.txt --epochs 30 --save out
```

여기서 `my_notes.txt`에는 아래 같은 자료가 잘 맞는다.

- 내 메모
- 작업 로그
- 개인 위키
- 자주 쓰는 표현
- 프로젝트 문서

가능하면 처음 코퍼스는 아래처럼 작게 시작하는 편이 좋다.

- 200~2,000줄 정도의 짧은 메모
- 하나의 프로젝트 문서 묶음
- 내 말투가 드러나는 대화 로그 일부

### 4. 생성 확인

```bash
python3 generate.py "오늘 해야 할 일은?" --model out/final_model.pt
```

이 단계에서 보는 건 “정답률”보다 아래다.

- 말투가 내 텍스트에 닮았는가
- 반복/붕괴가 심한가
- 생성 상태가 너무 발산하지 않는가
- 작은 프롬프트에 반응하는 흐름이 보이는가

## 사용자 유형별 활용법

### A. 개인 사용자

추천 목적:

- 나만의 말투 실험
- 개인 메모를 반영하는 작은 언어 코어
- 로컬 개인 assistant의 씨앗

추천 순서:

1. 메모/대화 로그를 텍스트로 모은다
2. `train.py`로 작은 모델을 만든다
3. `generate.py`로 생성 감각을 본다
4. `personal_memory.py`를 붙인다

### B. 연구자/개발자

추천 목적:

- ODE 기반 언어 모델 실험
- attention 대체 구조 연구
- memory / adaptation / rollback 구조 연구

추천 순서:

1. `dynllm/dynamics_core.py`부터 읽는다
2. `dynllm/memory.py`로 장기기억 구조를 본다
3. `dynllm/model.py`에서 전체 흐름을 본다
4. `tests/`로 현재 가정과 경계를 확인한다

### C. 제품/에이전트 개발자

추천 목적:

- 개인화된 private runtime
- memory-aware agent core
- teacher-distillation 기반 경량 엔진

추천 순서:

1. `train.py`, `generate.py`로 최소 런타임 확보
2. `personal_memory.py`로 사용자 기억 축적
3. `distill_bridge.py`로 teacher 응답 누적
4. 별도 커넥터 저장소에서 app/agent 연결

## 외부 LLM이 없어도 어떻게 쓰는가

이 질문이 이 엔진의 핵심입니다.

외부 teacher가 없어도 아래 흐름은 그대로 성립해야 한다.

1. 개인 코퍼스 수집
2. 작은 로컬 학습
3. 생성 확인
4. personal memory 누적
5. 주기적 재학습

즉 외부 LLM은 `있으면 더 빠르게 배우는 보조 수단`이지, `없으면 작동하지 않는 필수 인프라`가 아니다.

## 이 엔진이 주는 실제 가치

### 1. 개인화

기존 대형 LLM은 fine-tuning이 무겁다.  
이 엔진은 작은 코퍼스와 메모리 주입만으로도 “내 데이터 위에 반응하는 코어”를 만들려는 방향에 더 잘 맞는다.

### 2. 해석 가능성

핵심 흐름이 더 분해 가능하다.

- state
- damping
- context coupling
- working memory
- Hebbian memory
- episodic memory
- adaptation

즉 “왜 이런 출력이 나왔는가”를 더 나눠서 볼 수 있다.

### 3. 에이전트 코어 적합성

대화만 잘하는 LLM보다,

- 기억 누적
- 상태 전이
- drift 감시
- rollback

이 중요한 에이전트 코어 쪽에 더 잘 맞는다.

## 실제로 사용할 때 주의할 점

### 1. 지금은 char-level

아직 byte/subword tokenizer가 아니다.  
그래서 속도와 품질 한계가 있다.

### 2. 대규모 지식 모델이 아니다

이 엔진은 “세상 모든 지식”을 바로 가진 모델이 아니라,  
내가 직접 키우고 연결해 가는 코어다.

### 3. 온라인 적응은 장점이자 위험이다

적응이 너무 강하면 drift가 생긴다.  
그래서 TrustGate, RollbackPolicy가 중요하다.

### 4. first use는 작은 데이터가 좋다

처음부터 큰 코퍼스로 가기보다,

- 짧은 문장 집합
- 개인 메모 1개
- 프로젝트 위키 일부

처럼 작은 자료부터 시작하는 편이 낫다.

## 추천 사용 시나리오

### 시나리오 1: 개인 메모 LLM

- 입력: 내 메모/저널
- 목표: 내 표현과 관심사를 반영하는 작고 실험적인 언어 코어

### 시나리오 2: 연구용 동역학 언어 엔진

- 입력: 작은 benchmark 텍스트
- 목표: ODE + memory + adaptation 구조 검증

### 시나리오 3: 에이전트용 상태 기반 코어

- 입력: 이벤트 로그 + 사용자 히스토리
- 목표: 기억과 상태 전이를 가진 로컬 assistant

### 시나리오 4: 네트워크 없는 개인 대뇌피질

- 입력: 로컬 문서, 메모, 대화 기록
- 목표: 인터넷이나 외부 LLM 서비스 없이도 계속 성장하는 개인 언어 코어

## 다음에 무엇을 붙여야 하는가

우선순위는 이렇다.

1. byte/subword tokenizer
2. 더 안정한 train/generate 실험 결과 기록
3. personal memory 자동 적재기
4. distillation workflow
5. 외부 connector 저장소
6. long-context 평가

## 한 줄 정리

**이 엔진은 “바로 쓰는 범용 LLM”이 아니라, 내가 직접 키우고 개인화해 갈 수 있는 동역학 언어 엔진 베이스다.**
