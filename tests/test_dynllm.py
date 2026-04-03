"""Dynamical LLM Foundation — Phase A + B + C + D 테스트."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynllm.tokenizer import DynTokenizer, ByteTokenizer, load_tokenizer
from dynllm.state_encoder import StateEncoder
from dynllm.integrator import rk4_step, euler_step
from dynllm.dynamics_core import (
    DynamicsConfig, DynamicsCore, VectorField,
    ContextCoupling, ContextBuffer, TimescaleSeparator,
)
from dynllm.memory import (
    MemoryConfig, MemorySystem, WorkingMemory,
    HebbianMemory, EpisodicMemory,
)
from dynllm.readout import Readout, OnlineAdapter, ConsolidationScheduler
from dynllm.model import DynLLM, DynLLMConfig
from dynllm.stability import (
    clamp_state, detect_instability, TrustGate,
    SafeStateBuffer, VelocityMonitor, RollbackPolicy,
)


# ━━━━━━━━━━━━━━━━━━━━ Phase A ━━━━━━━━━━━━━━━━━━━━

class TestTokenizer:
    def test_encode_decode(self):
        tok = DynTokenizer().fit(["hello world"])
        ids = tok.encode("hello")
        assert tok.decode(ids) == "hello"

    def test_vocab_size(self):
        tok = DynTokenizer().fit(["abc"])
        assert tok.vocab_size >= 4 + 3


class TestStateEncoder:
    def test_shape(self):
        enc = StateEncoder(vocab_size=32, d_state=64)
        out = enc(torch.tensor([[1, 2, 3]]))
        assert out.shape == (1, 3, 64)


class TestIntegrator:
    def test_rk4_decay(self):
        x1 = rk4_step(lambda x, t: -0.5 * x, torch.tensor([1.0, 2.0]), torch.tensor(0.0), 0.1)
        assert x1[0].item() < 1.0

    def test_euler_vs_rk4(self):
        f = lambda x, t: -x
        x0, t = torch.ones(4), torch.tensor(0.0)
        exact = x0 * torch.exp(torch.tensor(-0.1))
        assert (rk4_step(f, x0, t, 0.1) - exact).abs().max() < (euler_step(f, x0, t, 0.1) - exact).abs().max()


# ━━━━━━━━━━━━━━━━━━━━ Phase B: Dynamics Core ━━━━━━━━━━━━━━━━━━━━

class TestDynamicsCore:
    def test_step_token(self):
        cfg = DynamicsConfig(d_state=32, d_input=32, d_memory=32, n_substeps=2, use_timescale_sep=False)
        core = DynamicsCore(cfg)
        x_next = core.step_token(torch.zeros(2, 32), torch.randn(2, 32), torch.zeros(2, 32))
        assert x_next.shape == (2, 32) and not torch.isnan(x_next).any()

    def test_damping_prevents_explosion(self):
        cfg = DynamicsConfig(d_state=16, d_input=16, d_memory=16, damping=1.0, use_timescale_sep=False)
        core = DynamicsCore(cfg)
        x, u, m = torch.randn(1, 16) * 5, torch.randn(1, 16), torch.zeros(1, 16)
        for _ in range(20):
            x = core.step_token(x, u, m)
        assert x.abs().max().item() < 50.0

    def test_step_with_context_buffer(self):
        cfg = DynamicsConfig(d_state=32, d_input=32, d_memory=32, n_substeps=2, context_window=8, use_timescale_sep=False)
        core = DynamicsCore(cfg)
        ctx_buf = ContextBuffer(8, 32)
        ctx_buf.init(2, torch.device("cpu"))
        x = torch.zeros(2, 32)
        for _ in range(5):
            x = core.step_token(x, torch.randn(2, 32), torch.zeros(2, 32), ctx_buf)
        assert x.shape == (2, 32)

    def test_step_with_timescale(self):
        cfg = DynamicsConfig(d_state=32, d_input=32, d_memory=32, n_substeps=2, use_timescale_sep=True, fast_ratio=0.5)
        x_next = DynamicsCore(cfg).step_token(torch.zeros(2, 32), torch.randn(2, 32), torch.zeros(2, 32))
        assert x_next.shape == (2, 32)

    def test_full_sequence(self):
        cfg = DynamicsConfig(d_state=32, d_input=32, d_memory=32, n_substeps=2, context_window=4, use_timescale_sep=True)
        states = DynamicsCore(cfg)(torch.zeros(2, 32), torch.randn(2, 10, 32))
        assert states.shape == (2, 10, 32)


class TestContextCoupling:
    def test_shape(self):
        out = ContextCoupling(32, 4, 8)(torch.randn(2, 32), torch.randn(2, 8, 32))
        assert out.shape == (2, 32)

    def test_gradient_flow(self):
        cc = ContextCoupling(16, 2, 4)
        x = torch.randn(1, 16, requires_grad=True)
        cc(x, torch.randn(1, 4, 16)).sum().backward()
        assert x.grad is not None


class TestContextBuffer:
    def test_push_and_get(self):
        buf = ContextBuffer(4, 16)
        buf.init(2, torch.device("cpu"))
        for i in range(6):
            buf.push(torch.ones(2, 16) * i)
        h = buf.get()
        assert h.shape == (2, 4, 16)
        assert (h[:, -1, 0] == 5.0).all()


class TestTimescaleSeparator:
    def test_split_merge(self):
        ts = TimescaleSeparator(32, 0.5)
        x = torch.randn(2, 32)
        fast, slow = ts.split(x)
        assert fast.shape == (2, 16) and slow.shape == (2, 16)
        assert ts.merge(fast, slow).shape == (2, 32)


class TestVectorField:
    def test_with_context(self):
        cfg = DynamicsConfig(d_state=32, d_input=32, d_memory=32)
        dx = VectorField(cfg)(torch.randn(2, 32), torch.randn(2, 32), torch.randn(2, 32), torch.randn(2, 32))
        assert dx.shape == (2, 32)

    def test_without_context(self):
        cfg = DynamicsConfig(d_state=32, d_input=32, d_memory=32)
        dx = VectorField(cfg)(torch.randn(2, 32), torch.randn(2, 32), torch.zeros(2, 32))
        assert dx.shape == (2, 32)


# ━━━━━━━━━━━━━━━━━━━━ Phase C: Memory ━━━━━━━━━━━━━━━━━━━━

class TestWorkingMemory:
    def test_read_write(self):
        cfg = MemoryConfig(d_state=32, n_working_slots=4)
        wm = WorkingMemory(cfg)
        slots, acts = wm.init_state(2, torch.device("cpu"))
        x = torch.randn(2, 32)
        readout = wm.read(x, slots, acts)
        assert readout.shape == (2, 32)
        new_s, new_a = wm.write(x, slots, acts)
        assert new_s.shape == slots.shape

    def test_summary_feedback(self):
        cfg = MemoryConfig(d_state=32, n_working_slots=4)
        wm = WorkingMemory(cfg)
        slots, acts = wm.init_state(2, torch.device("cpu"))
        summary = wm.summary(slots, acts)
        assert summary.shape == (2, 32)


class TestHebbianMemory:
    def test_store_recall(self):
        cfg = MemoryConfig(d_state=16)
        heb = HebbianMemory(cfg)
        heb.store(torch.randn(16))
        assert heb.n_stored.item() == 1
        recalled = heb.recall(torch.randn(1, 16))
        assert recalled.shape == (1, 16)

    def test_selective_recall(self):
        cfg = MemoryConfig(d_state=16, selective_top_k=2)
        heb = HebbianMemory(cfg)
        for _ in range(5):
            heb.store(torch.randn(16))
        result = heb.selective_recall(torch.randn(1, 16))
        assert result.shape == (1, 16)

    def test_selective_recall_empty(self):
        cfg = MemoryConfig(d_state=16, selective_top_k=2)
        heb = HebbianMemory(cfg)
        result = heb.selective_recall(torch.randn(1, 16))
        assert result.shape == (1, 16)

    def test_batch_selective_recall(self):
        cfg = MemoryConfig(d_state=16, selective_top_k=2)
        heb = HebbianMemory(cfg)
        for _ in range(5):
            heb.store(torch.randn(16))
        result = heb.selective_recall(torch.randn(3, 16))
        assert result.shape == (3, 16)


class TestEpisodicMemory:
    def test_store_and_recall(self):
        cfg = MemoryConfig(d_state=32, n_episodes=8)
        ep = EpisodicMemory(cfg)
        traj = torch.randn(10, 32)
        ep.store_episode(traj)
        assert ep.n_stored.item() == 1
        result = ep.recall_episode(torch.randn(2, 32))
        assert result.shape == (2, 32)

    def test_store_batch(self):
        cfg = MemoryConfig(d_state=32, n_episodes=8)
        ep = EpisodicMemory(cfg)
        batch_traj = torch.randn(3, 10, 32)
        ep.store_episode(batch_traj)
        assert ep.n_stored.item() == 3

    def test_recall_empty(self):
        cfg = MemoryConfig(d_state=32, n_episodes=8)
        ep = EpisodicMemory(cfg)
        result = ep.recall_episode(torch.randn(1, 32))
        assert result.shape == (1, 32)


class TestMemorySystem:
    def test_full_step(self):
        cfg = MemoryConfig(d_state=32, n_working_slots=4)
        mem = MemorySystem(cfg)
        slots, acts = mem.init_state(2, torch.device("cpu"))
        x = torch.randn(2, 32)
        readout, slots, acts = mem.step(x, slots, acts)
        assert readout.shape == (2, 32)

    def test_wm_feedback(self):
        cfg = MemoryConfig(d_state=32, n_working_slots=4)
        mem = MemorySystem(cfg)
        slots, acts = mem.init_state(2, torch.device("cpu"))
        feedback = mem.wm_feedback(slots, acts)
        assert feedback.shape == (2, 32)

    def test_store_episode(self):
        cfg = MemoryConfig(d_state=32)
        mem = MemorySystem(cfg)
        mem.store_episode(torch.randn(10, 32))
        assert mem.episodic.n_stored.item() == 1


# ━━━━━━━━━━━━━━━━━━━━ Phase D: Readout & Adaptation ━━━━━━━━━━━━━━━━━━━━

class TestReadout:
    def test_shape(self):
        logits = Readout(32, 100)(torch.randn(2, 10, 32))
        assert logits.shape == (2, 10, 100)


class TestOnlineAdapter:
    def test_adapt(self):
        adapter = OnlineAdapter(16, 32)
        info = adapter.adapt(torch.randn(16), target_id=5, logits=torch.randn(32))
        assert "trust" in info and "loss" in info

    def test_fast_decay(self):
        adapter = OnlineAdapter(16, 32, fast_decay=0.9)
        adapter.fast_weight.fill_(1.0)
        adapter.adapt(torch.randn(16), target_id=0, logits=torch.randn(32))
        assert adapter.fast_weight.abs().max().item() < 1.5

    def test_state_adapter(self):
        adapter = OnlineAdapter(16, 32)
        state = torch.randn(1, 16)
        target = torch.randn(1, 16)
        adapter.adapt_state(state, target)
        assert adapter.state_adapter.abs().max().item() > 0

    def test_consolidate(self):
        adapter = OnlineAdapter(16, 32)
        readout = Readout(16, 32)
        adapter.fast_weight.fill_(0.5)
        adapter.consolidate(readout, ratio=0.1)
        assert readout.proj.weight.abs().max().item() > 0

    def test_adaptation_summary(self):
        adapter = OnlineAdapter(16, 32)
        summary = adapter.adaptation_summary
        assert "n_updates" in summary and "fast_weight_norm" in summary


class TestConsolidationScheduler:
    def test_periodic_consolidation(self):
        adapter = OnlineAdapter(16, 32)
        readout = Readout(16, 32)
        sched = ConsolidationScheduler(interval=5)

        adapter.fast_weight.fill_(1.0)
        adapter.n_updates.fill_(10)

        consolidated = False
        for _ in range(10):
            if sched.step(adapter, readout):
                consolidated = True
        assert consolidated

    def test_stats(self):
        sched = ConsolidationScheduler(interval=5)
        assert sched.stats["total_steps"] == 0


# ━━━━━━━━━━━━━━━━━━━━ Phase D: Stability ━━━━━━━━━━━━━━━━━━━━

class TestStability:
    def test_clamp(self):
        clamped = clamp_state(torch.randn(2, 32) * 100, max_norm=5.0)
        assert clamped.norm(dim=-1).max().item() <= 5.1

    def test_detect_instability(self):
        assert not detect_instability(torch.randn(2, 32))
        assert detect_instability(torch.ones(2, 32) * 200)

    def test_trust_gate_ema(self):
        tg = TrustGate()
        for _ in range(50):
            tg.update(1.0)
        assert tg.trust > 0
        assert tg.ema_loss > 0
        diag = tg.diagnostics
        assert "ema_loss" in diag and diag["step_count"] == 50

    def test_trust_gate_warmup(self):
        tg = TrustGate()
        trust_early = tg.update(1.0)
        assert trust_early < 0.2

    def test_safe_state_buffer(self):
        buf = SafeStateBuffer(max_history=3)
        buf.save(torch.ones(1, 16))
        buf.save(torch.ones(1, 16) * 2)
        rolled = buf.rollback(torch.ones(1, 16) * 999)
        assert torch.allclose(rolled, torch.ones(1, 16) * 2)

    def test_interpolated_rollback(self):
        buf = SafeStateBuffer(max_history=5)
        buf.save(torch.ones(1, 16) * 1.0)
        buf.save(torch.ones(1, 16) * 3.0)
        interp = buf.interpolated_rollback(torch.zeros(1, 16), alpha=0.5)
        expected = 0.5 * 3.0 + 0.5 * 1.0
        assert torch.allclose(interp, torch.ones(1, 16) * expected)

    def test_velocity_monitor(self):
        vm = VelocityMonitor()
        info = vm.record(torch.zeros(1, 16), torch.ones(1, 16))
        assert "velocity" in info


class TestRollbackPolicy:
    def test_normal_state(self):
        policy = RollbackPolicy()
        buf = SafeStateBuffer()
        x = torch.randn(1, 16)
        x_out, info = policy.check_and_rollback(x, buf)
        assert not info["rolled_back"]

    def test_single_instability(self):
        policy = RollbackPolicy()
        buf = SafeStateBuffer()
        buf.save(torch.ones(1, 16))
        x_bad = torch.ones(1, 16) * 200
        x_out, info = policy.check_and_rollback(x_bad, buf)
        assert info["rolled_back"]
        assert info["mode"] == "simple_rollback"

    def test_consecutive_instability(self):
        policy = RollbackPolicy(consecutive_limit=2)
        buf = SafeStateBuffer()
        buf.save(torch.ones(1, 16))
        buf.save(torch.ones(1, 16) * 0.5)

        x_bad = torch.ones(1, 16) * 200
        policy.check_and_rollback(x_bad, buf)
        _, info = policy.check_and_rollback(x_bad, buf)
        assert info["mode"] == "cooldown_triggered"
        assert policy.in_cooldown

    def test_diagnostics(self):
        policy = RollbackPolicy()
        diag = policy.diagnostics
        assert "total_rollbacks" in diag


# ━━━━━━━━━━━━━━━━━━━━ Integration ━━━━━━━━━━━━━━━━━━━━

class TestDynLLM:
    def test_forward(self):
        cfg = DynLLMConfig(vocab_size=32, d_state=32, use_memory=False, use_timescale_sep=False)
        logits = DynLLM(cfg)(torch.randint(1, 32, (2, 10)))
        assert logits.shape == (2, 10, 32)

    def test_forward_with_memory(self):
        cfg = DynLLMConfig(vocab_size=32, d_state=32, use_memory=True, use_timescale_sep=False)
        logits = DynLLM(cfg)(torch.randint(1, 32, (2, 10)))
        assert logits.shape == (2, 10, 32)

    def test_forward_phase_b_full(self):
        cfg = DynLLMConfig(vocab_size=32, d_state=32, use_memory=True,
                           use_timescale_sep=True, context_window=8, context_heads=2)
        logits = DynLLM(cfg)(torch.randint(1, 32, (2, 12)))
        assert logits.shape == (2, 12, 32)

    def test_forward_phase_cd_full(self):
        cfg = DynLLMConfig(
            vocab_size=32, d_state=32, use_memory=True,
            use_timescale_sep=True, context_window=4,
            context_heads=2, n_episodes=4, selective_top_k=2,
        )
        model = DynLLM(cfg)
        model.train()
        logits = model(torch.randint(1, 32, (2, 8)))
        assert logits.shape == (2, 8, 32)

    def test_generate(self):
        cfg = DynLLMConfig(vocab_size=32, d_state=32, use_memory=False, use_timescale_sep=False)
        gen = DynLLM(cfg).generate([1, 5, 10], max_tokens=10)
        assert isinstance(gen, list) and len(gen) <= 10

    def test_generate_phase_cd(self):
        cfg = DynLLMConfig(vocab_size=32, d_state=32, use_memory=True,
                           use_timescale_sep=True, context_window=4, n_episodes=4)
        gen = DynLLM(cfg).generate([1, 5, 10], max_tokens=10)
        assert isinstance(gen, list)

    def test_backward(self):
        cfg = DynLLMConfig(vocab_size=16, d_state=16, use_memory=False,
                           n_substeps=1, use_timescale_sep=False)
        model = DynLLM(cfg)
        logits = model(torch.randint(1, 16, (2, 5)))
        loss = F.cross_entropy(logits.reshape(-1, 16), torch.randint(0, 16, (2, 5)).reshape(-1))
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_backward_phase_cd(self):
        cfg = DynLLMConfig(
            vocab_size=16, d_state=16, use_memory=True, n_substeps=1,
            use_timescale_sep=True, context_window=4, context_heads=2,
            n_episodes=4,
        )
        model = DynLLM(cfg)
        model.train()
        logits = model(torch.randint(1, 16, (2, 6)))
        loss = F.cross_entropy(logits.reshape(-1, 16), torch.randint(0, 16, (2, 6)).reshape(-1))
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_param_count(self):
        cfg = DynLLMConfig(vocab_size=256, d_state=128)
        n = DynLLM(cfg).count_parameters()
        assert n > 0
        print(f"  DynLLM Phase C+D params: {n:,}")


# ━━━━━━━━━━━━━━━━━━━━ Phase E: Personal Memory & Distill ━━━━━━━━━━━━━━━━━━━━

class TestPersonalMemory:
    def test_log_and_recall(self, tmp_path):
        from dynllm.personal_memory import PersonalMemoryStore, Interaction
        store = PersonalMemoryStore(db_path=tmp_path / "test.db")
        store.log(Interaction(user_input="hello", response="world"))
        store.log(Interaction(user_input="foo", response="bar"))
        recent = store.recent_interactions(limit=5)
        assert len(recent) == 2
        stats = store.stats()
        assert stats["n_interactions"] == 2
        store.close()

    def test_crystal_store(self, tmp_path):
        from dynllm.personal_memory import PersonalMemoryStore
        store = PersonalMemoryStore(db_path=tmp_path / "test2.db")
        vec = torch.randn(32)
        store.store_crystal("test pattern", vec, importance=0.8)
        crystals = store.recall_crystals(top_k=5)
        assert len(crystals) == 1
        assert crystals[0]["importance"] == 0.8
        store.close()


class TestDistillBridge:
    def test_buffer_add_and_stats(self):
        from dynllm.distill_bridge import DistillBuffer, DistillSample
        buf = DistillBuffer()
        buf.add(DistillSample(input_text="q1", target_text="a1", confidence=0.8))
        buf.add(DistillSample(input_text="q2", target_text="a2", confidence=0.2))
        assert buf.size == 2
        texts = buf.get_training_texts(min_confidence=0.5)
        assert len(texts) == 1

    def test_buffer_save_load(self, tmp_path):
        from dynllm.distill_bridge import DistillBuffer, DistillSample
        buf = DistillBuffer()
        buf.add(DistillSample(input_text="hello", target_text="world"))
        path = tmp_path / "distill.jsonl"
        buf.save(path)

        buf2 = DistillBuffer()
        buf2.load(path)
        assert buf2.size == 1

    def test_bridge_without_connector(self):
        from dynllm.distill_bridge import DistillBridge
        bridge = DistillBridge(library_connector=None)
        text, conf = bridge.query("test")
        assert conf == 0.0
        assert bridge.status()["connector"] == "None"


class TestEvaluate:
    def test_perplexity(self):
        from dynllm.evaluate import compute_perplexity
        tok = DynTokenizer().fit(["abcabc" * 20])
        cfg = DynLLMConfig(vocab_size=tok.vocab_size, d_state=32,
                           use_memory=False, use_timescale_sep=False)
        model = DynLLM(cfg)
        ppl = compute_perplexity(model, "abcabc" * 10, tok, seq_len=16)
        assert ppl > 0 and ppl < 1e6

    def test_memory_utilization(self):
        from dynllm.evaluate import compute_memory_utilization
        cfg = DynLLMConfig(vocab_size=32, d_state=32, use_memory=True,
                           use_timescale_sep=False)
        model = DynLLM(cfg)
        util = compute_memory_utilization(model)
        assert util["has_memory"]
        assert util["hebbian_utilization"] == 0.0

    def test_memory_recall_rate(self):
        from dynllm.evaluate import compute_memory_recall_rate
        cfg = DynLLMConfig(vocab_size=32, d_state=16, use_memory=True,
                           use_timescale_sep=False)
        model = DynLLM(cfg)

        target = torch.randn(16)
        model.memory.hebbian.store(target)
        model.memory.episodic.store_episode(target.unsqueeze(0).repeat(4, 1))

        report = compute_memory_recall_rate(model, cues=[target], targets=[target])
        assert report["has_memory"]
        assert report["n_trials"] == 1
        assert 0.0 <= report["overall_recall_rate_0_1"] <= 1.0


# ━━━━━━━━━━━━━━ ByteTokenizer ━━━━━━━━━━━━━━

class TestByteTokenizer:
    def test_encode_decode_ascii(self):
        tok = ByteTokenizer()
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == 1  # BOS
        assert ids[-1] == 2  # EOS
        assert tok.decode(ids) == "hello"

    def test_encode_decode_korean(self):
        tok = ByteTokenizer()
        text = "안녕하세요"
        ids = tok.encode(text, add_bos=False, add_eos=False)
        assert len(ids) == len(text.encode("utf-8"))  # 15 bytes
        assert tok.decode(ids) == text

    def test_fixed_vocab(self):
        tok = ByteTokenizer()
        assert tok.vocab_size == 260  # 256 + 4 special

    def test_no_fit_needed(self):
        tok = ByteTokenizer()
        same = tok.fit(["anything"])
        assert same is tok
        assert tok.vocab_size == 260

    def test_save_load(self, tmp_path):
        tok = ByteTokenizer()
        path = tmp_path / "tok.json"
        tok.save(path)
        loaded = ByteTokenizer.load(path)
        assert loaded.vocab_size == 260
        assert loaded.decode(loaded.encode("test")) == "test"

    def test_unified_loader_byte(self, tmp_path):
        tok = ByteTokenizer()
        path = tmp_path / "tok.json"
        tok.save(path)
        loaded = load_tokenizer(path)
        assert isinstance(loaded, ByteTokenizer)

    def test_unified_loader_char(self, tmp_path):
        tok = DynTokenizer().fit(["abc"])
        path = tmp_path / "tok.json"
        tok.save(path)
        loaded = load_tokenizer(path)
        assert isinstance(loaded, DynTokenizer)

    def test_mode_property(self):
        assert ByteTokenizer().mode == "byte"
        assert DynTokenizer().fit(["a"]).mode == "char"

    def test_model_with_byte_tokenizer(self):
        tok = ByteTokenizer()
        cfg = DynLLMConfig(vocab_size=tok.vocab_size, d_state=32, use_memory=False)
        model = DynLLM(cfg)
        text = "hello 안녕"
        ids = tok.encode(text, add_bos=False, add_eos=False)
        x = torch.tensor([ids])
        logits = model(x)
        assert logits.shape == (1, len(ids), tok.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
