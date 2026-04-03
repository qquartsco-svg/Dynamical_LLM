"""
Dynamical LLM Foundation — dynllm
v0.5.0 (Phase A–F base implementation)

대뇌피질형 동역학 언어 엔진.
Transformer 대신 ODE 상태 진화로 시퀀스를 처리한다.

핵심 수식: ẋ = −λx + F(x, u, m; θ)

포함 레이어:
  L0  tokenizer          char-level 토크나이저
  L1  state_encoder      토큰 → 초기 상태벡터
  L2  dynamics_core      ODE 코어 (ContextCoupling, TimescaleSeparator)
  L3  memory             4단 기억 (단기·작업·헤비안·에피소딕)
  L4  readout            상태 → logits
  L5  stability          TrustGate, VelocityMonitor, RollbackPolicy
  --  personal_memory    PersonalMemoryStore, MemoryInjector (Phase E)
  --  distill_bridge     DistillBuffer, DistillBridge (Phase E)
  --  evaluate           perplexity, diversity, memory_utilization (Phase E)
  --  system_bridge      DynLLM <-> Atom/Athena/Aton/Pharaoh contracts
"""
__version__ = "0.5.0"
__codename__ = "Phase A-F — Foundation + Personalization + Governance"
