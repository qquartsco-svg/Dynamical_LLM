> **English.** Korean (정본): [README.md](README.md)

# Dynamical LLM Foundation

**Version:** `v0.5.0` — Phase A–E base implementation

This is not a Transformer clone.  
Each incoming token drives the internal state forward via an ODE, coupled to a 4-tier memory hierarchy, with online adaptation built in — a **foundational layer for dynamical language modeling**.

---

## Core Equation

```text
ẋ = -λx + F(x, u, m; θ)
```

| Symbol | Meaning |
|--------|---------|
| `x` | Hidden state vector |
| `λ` | Learnable damping (prevents divergence) |
| `u` | Input impulse from current token |
| `m` | Memory readout signal (working + Hebbian) |
| `θ` | Learnable vector field parameters |

---

## Architecture — 6 Layers

```
L0  Token Interface      char → integer ID
L1  State Encoder        ID → initial state vector x₀
L2  Dynamics Core        ODE evolution: context coupling + time-scale separation + refined gating
L3  Memory               short-term (hidden x) · working (PFC) · Hebbian (long-term) · episodic
L4  Readout              state → vocabulary logits
L5  Online Adaptation    fast/slow weights + trust gate + consolidation scheduler
```

---

## Completed Phases

| Phase | Focus | Key Deliverables |
|-------|-------|-----------------|
| A | Minimal working model | `train.py`, `generate.py`, SafeStateBuffer, per-sample Hebbian storage |
| B | Learnable dynamics | ContextCoupling, TimescaleSeparator, VelocityMonitor, spectral normalization |
| C | Memory enhancement | SelectiveRecall, EpisodicMemory, bidirectional WM ↔ Dynamics feedback |
| D | Online adaptation | FastWeightDecay, StateAdapter, ConsolidationScheduler, RollbackPolicy |
| E | Personalization + external bridge readiness | PersonalMemoryStore, MemoryInjector, DistillBridge |

---

## File Layout

```
Dynamical_LLM_Foundation/
├── dynllm/
│   ├── tokenizer.py          # L0
│   ├── state_encoder.py      # L1
│   ├── dynamics_core.py      # L2 — ContextCoupling, TimescaleSeparator
│   ├── memory.py             # L3 — WorkingMem, HebbianMem, EpisodicMem
│   ├── readout.py            # L4+L5 — OnlineAdapter, ConsolidationScheduler
│   ├── stability.py          # TrustGate, VelocityMonitor, RollbackPolicy
│   ├── model.py              # Integrated DynLLM
│   ├── personal_memory.py    # Phase E — PersonalMemoryStore, MemoryInjector
│   ├── distill_bridge.py     # Phase E — DistillBuffer, DistillBridge
│   └── evaluate.py           # Phase E — perplexity, diversity, memory utilization
├── train.py
├── generate.py
└── tests/ (64 tests)
```

---

## Quick Start

```bash
pip install -e .

# Train (default)
python3 train.py --epochs 20

# Train from a corpus file
python3 train.py --corpus mytext.txt --epochs 50 --save ckpt.pt

# Generate
python3 generate.py "hello world"
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
logits = model(ids)         # [2, 32, 128]
```

---

## Commercial Position

The strongest product direction here is not “replace a large general-purpose LLM outright”.
It is closer to:

- an on-device personal language engine base
- an adaptive agent core
- a private LLM runtime with explicit memory/adaptation/drift controls
- a bridge from research-grade dynamical architectures to productizable cores

Its current value therefore leans more toward personalization, interpretability, and controllable dynamics than raw benchmark competition.

---

## External Bridge Readiness

Phase E already includes `personal_memory.py` and `distill_bridge.py`,
so the repository is ready to connect personal memory and teacher-style distillation flows.
However, `Atom.connectors.dynllm_connector` itself is not shipped inside this repository.

```python
from dynllm.personal_memory import PersonalMemoryStore
from dynllm.distill_bridge import DistillBuffer, DistillSample

store = PersonalMemoryStore(":memory:")
buffer = DistillBuffer()

store.log_interaction("How are you?", "I am doing fairly well.")
buffer.add(DistillSample("How are you?", "I am doing fairly well."))
```

---

## Validation

```bash
python3 -m pytest tests/ -q
# full runtime tests on a torch-enabled environment
# torch-dependent tests may be skipped if torch is unavailable

python3 scripts/release_check.py
```

Test categories: tokenizer, state encoder, integrator, dynamics core, memory tiers, stability, readout, online adapter, personal memory, distill bridge, evaluate, train/generate smoke entrypoints, package integrity.

---

## Limits

- Not designed to match large-scale pretrained Transformer performance
- Character-level tokenizer only — subword upgrade is post-Phase-E
- Long-horizon drift control is partially addressed via TrustGate + RollbackPolicy
- Large corpus training not yet benchmarked
- Current product value is closer to a personalizable dynamics-based language engine base than a finished commercial LLM

---

## Integrity Signature

```bash
python3 scripts/generate_signature.py   # regenerate SIGNATURE.sha256
python3 scripts/verify_signature.py     # verify
```

Details: [BLOCKCHAIN_INFO_EN.md](BLOCKCHAIN_INFO_EN.md)
