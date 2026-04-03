> **English.** Korean (정본): [README.md](README.md)

# Dynamical LLM Foundation

**Version:** `v0.6.0` — Phase A–G (Engine Hub Integration)

### v0.6.0 Update (Phase G)

> **4 Engine Hub integrations + 19 new tests (100 passed total)**

| New Module | Connected Engine | Capability |
|-----------|-----------------|-----------|
| `memory_rank_adapter.py` | MemoryRank (Cognitive_Kernel v1.1.0) | **PageRank memory re-ranking** — cosine candidates → Personalized PageRank "Googling" style memory search |
| `diagnostics.py` | ConvergenceDynamics (40_SPATIAL) | **Training convergence diagnostics** — convergence order, Lyapunov estimate, stability verdict |
| `diagnostics.py` | StatMech (40_SPATIAL) | **Output entropy analysis** — Gibbs entropy, diversity/temperature estimation |
| `diagnostics.py` | IIT (50_DIAGNOSTIC) | **Integrated information Φ approximation** — spectral gap, effective rank, coupling assessment |

Existing file changes:
- `memory.py`: added `ranked_selective_recall()` + `get_pattern_dict()`, MemoryGraph option in `MemorySystem`
- `personal_memory.py`: added `memory_links` table + `recall_crystals_ranked()` (PageRank) + `bump_access_count()`
- `model.py`: added `use_memory_rank` / `use_diagnostics` config options + `run_diagnostics()` method
- Validation: **100 passed** (81→100), signature 41 files matched

---

> The repository starts from the familiar `LLM` label, but its structural identity is closer to a `DLM (Dynamical Language Model)`.  
> In other words, the entry point is LLM, while the actual core is a dynamics-driven language system built around state evolution, memory, and adaptation.

Documents:
- [User Guide](USER_GUIDE.md)
  - who should use it and how
- [Roadmap](ROADMAP.md)
  - what should be built next
- [Offline Playbook](OFFLINE_PERSONAL_LLM_PLAYBOOK.md)
  - how to operate it without external LLM services
- [System Connection Map](SYSTEM_CONNECTION_MAP.md)
  - boundaries across Atom / Athena / Aton / Pharaoh / User
- [Experiment Log](EXPERIMENT_LOG.md)
  - training conditions, sample outputs, drift observations

This is not a Transformer clone.  
Each incoming token drives the internal state forward via an ODE, coupled to a 4-tier memory hierarchy, with online adaptation built in — a **foundational layer for dynamical language modeling**.

The core philosophy is an `offline-survivable personal cortical core`.
Large external LLMs may appear as optional teachers, but the engine should remain meaningful even when no network LLM service is available.

In one sentence:

**The main goal is to build a language core that can keep growing from personal corpus and personal memory even when no external LLM service is available.**

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

In short, this engine **does use deep learning**.  
The difference is that it does not learn through a deep Transformer attention stack; it learns through a **trainable dynamical function plus memory hierarchy** that evolves internal state over time.

- deep learning: update `θ` via gradient descent
- language-model training: minimize next-token prediction loss
- dynamical distinction: `token -> state impulse -> ODE evolution -> memory feedback -> next token`

---

## Architecture — 6 Layers

```
L0  Token Interface      char/byte → integer ID (char or byte mode)
L1  State Encoder        ID → initial state vector x₀
L2  Dynamics Core        ODE evolution: context coupling + time-scale separation + refined gating
L3  Memory               short-term (hidden x) · working (PFC) · Hebbian (long-term+PageRank) · episodic
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
| F | System governance linkage | `system_bridge.py`, boundary contracts for Atom/Athena/Aton/Pharaoh |
| **G** | **Engine Hub integration** | **MemoryGraph (PageRank), ConvergenceMonitor, EntropyAnalyzer, IntegrationDiagnostic** |

This table describes the `base implementation` state.  
The layer scaffolding is present, but practical maturity is still lower in tokenizer quality, training evidence, connectors, and benchmarks.

---

## File Layout

```
Dynamical_LLM_Foundation/
├── dynllm/
│   ├── tokenizer.py          # L0 — DynTokenizer (char) + ByteTokenizer (byte)
│   ├── state_encoder.py      # L1
│   ├── dynamics_core.py      # L2 — ContextCoupling, TimescaleSeparator
│   ├── memory.py             # L3 — WorkingMem, HebbianMem, EpisodicMem
│   ├── readout.py            # L4+L5 — OnlineAdapter, ConsolidationScheduler
│   ├── stability.py          # TrustGate, VelocityMonitor, RollbackPolicy
│   ├── model.py              # Integrated DynLLM
│   ├── personal_memory.py    # Phase E — PersonalMemoryStore, MemoryInjector
│   ├── distill_bridge.py     # Phase E — DistillBuffer, DistillBridge (optional teacher path)
│   ├── evaluate.py           # Phase E — perplexity, diversity, memory utilization
│   ├── system_bridge.py      # Phase F — DynLLM ↔ Atom/Athena/Aton/Pharaoh governance contracts
│   ├── memory_rank_adapter.py # Phase G — MemoryGraph (PageRank memory re-ranking)
│   └── diagnostics.py        # Phase G — Convergence/Entropy/Integration diagnostics
├── train.py
├── generate.py
├── examples/
│   └── run_dlm.py
└── tests/
    ├── test_dynllm.py
    ├── test_smoke_entrypoints.py
    ├── test_system_bridge.py
    └── test_package_integrity.py
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

# Minimal end-to-end example
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

## Offline Personal LLM Strategy

This engine should be read in two modes:

- `teacher-assisted`
  - an external LLM or internal library may provide distillation examples
- `offline-first`
  - the model grows from personal corpus, personal memory, and repeated local training only

Important points:

- `distill_bridge.py` is optional
- the primary path is `train.py + personal_memory.py + generate.py`
- the engine should remain usable even with no external teacher
- its strongest product value is a personal language core, not an API wrapper

---

## External Bridge Readiness

Phase E already includes `personal_memory.py` and `distill_bridge.py`,
so the repository is ready to connect personal memory and optional teacher-style distillation flows.
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

Test categories: tokenizer, state encoder, integrator, dynamics core, memory tiers, stability, readout, online adapter, personal memory, distill bridge, evaluate, memory recall rate, train/generate smoke entrypoints, system bridge, package integrity.

- `memory recall rate`
  - a rough proxy for how well injected memory is brought back into generation
- `system bridge / governance contracts`
  - boundary validation for draft text, confidence, memory sources, and risk tags

Verification numbers should be read by environment:

| Scope | Command | Meaning | Current result |
|------|------|------|-----------|
| torch full suite | internal pytest inside `release_check.py` | full runtime suite on torch-enabled environment | **100 passed** |
| release gate | `python3 scripts/release_check.py` | package identity + signature + tests | `OK` |
| signature | `python3 scripts/verify_signature.py` | 41-file hash verification | `passed=41 failed=0 missing=0` |

Tests requiring `torch` may be skipped in environments where it is not installed.

---

## Tokenizers

Two modes are available:

| Mode | Vocab | Multilingual | Notes |
|------|-------|-------------|-------|
| `DynTokenizer` (char) | corpus-dependent | requires `fit()` | simple, transparent |
| `ByteTokenizer` (byte) | fixed 260 | immediate, all languages | no re-fitting needed |

```bash
# byte-level training
python3 train.py --byte --epochs 20

# byte-level generation
python3 generate.py --byte "hello world"
```

---

## Experiment Summary

See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for full records.

| Exp | Tokenizer | Corpus | Loss drop | Final ppl | Time (CPU) |
|-----|-----------|--------|-----------|-----------|------------|
| 1 | char(22) | EN 630 chars | 0.54 | 9.1 | 90s |
| 2 | byte(260) | EN 630 chars | 1.26 | 7.5 | 77s |
| 3 | byte(260) | KR 1845 bytes | 2.23 | 2.5 | 239s |

Key observations:
- The ODE dynamics core **does learn** — loss consistently decreases across all experiments
- Damping (λ) **self-adapts** — reaches 0.44 for Korean (model finds its own stability point)
- ByteTokenizer converges **2.3× faster** than char
- Inference speed: **~1,170 tok/s** (CPU, memory off) / **~950 tok/s** (memory on)

---

## Limits

- Not designed to match large-scale pretrained Transformer performance
- Long-horizon drift control is partially addressed via TrustGate + RollbackPolicy
- Large corpus training not yet benchmarked
- Byte-level Korean requires more epochs (learning valid UTF-8 3-byte sequences)
- Current product value is closer to a personalizable dynamics-based language engine base than a finished commercial LLM

---

## Integrity Signature

```bash
python3 scripts/generate_signature.py   # regenerate SIGNATURE.sha256
python3 scripts/verify_signature.py     # verify
```

Details: [BLOCKCHAIN_INFO_EN.md](BLOCKCHAIN_INFO_EN.md)

## Tokenizer Usage Notes

The project currently supports two tokenizer modes:

- `DynTokenizer`
  - corpus-dependent char-level tokenizer
- `ByteTokenizer`
  - fixed 260-vocab byte-level tokenizer

Suggested starting points:

- small personal corpus + char-level: `20-50 epochs`
- byte-level Korean: usually needs longer runs than char-level, so start by observing `30-80 epochs`
