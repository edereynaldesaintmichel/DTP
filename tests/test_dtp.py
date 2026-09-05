"""CPU tests on a tiny random-weight Qwen3 config.

Run: .venv/bin/python -m pytest tests -q
"""

import math

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dtp.model import DTPCache, DTPQwen3

torch.manual_seed(0)

B, S, V = 2, 32, 512


def tiny_model():
    # 8 q heads / 4 kv heads: L=2 and L=4 both hit the sharded-KV path with a
    # GQA expand, like the real Qwen3-0.6B (16/8). head_dim=16 keeps Qwen3's
    # num_heads * head_dim != hidden_size property (128 vs 64).
    cfg = Qwen3Config(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=256,
    )
    torch.manual_seed(7)
    m = Qwen3ForCausalLM(cfg)
    m.eval()
    return m


def ids(bsz=B, seq=S):
    torch.manual_seed(3)
    return torch.randint(0, V, (bsz, seq))


def test_l1_delta0_matches_hf():
    hf = tiny_model()
    x = ids()
    ref = hf(x).logits
    dtp = DTPQwen3(hf, n_devices=1, delta=0)
    got = dtp(x).logits
    assert torch.allclose(ref, got, atol=2e-4, rtol=1e-4), (ref - got).abs().max()


def test_shard_partials_sum_to_full():
    hf = tiny_model()
    d1 = DTPQwen3(hf, n_devices=1, delta=0)
    d4 = DTPQwen3(hf, n_devices=4, delta=0)
    layer = hf.model.layers[1]
    torch.manual_seed(11)
    h = torch.randn(1, B, S, 64)
    pos_ids = torch.arange(S)[None].expand(B, S)
    pe = hf.model.rotary_emb(h, pos_ids)
    mask, is_causal = DTPQwen3._build_mask(S, 0, h.device)

    a1 = d1._attn_partials(layer, h, pe, mask, is_causal)
    a4 = d4._attn_partials(layer, h.expand(4, B, S, 64), pe, mask, is_causal)
    assert torch.allclose(a4.sum(0), a1[0], atol=1e-4), (a4.sum(0) - a1[0]).abs().max()

    m1 = d1._mlp_partials(layer, h)
    m4 = d4._mlp_partials(layer, h.expand(4, B, S, 64))
    assert torch.allclose(m4.sum(0), m1[0], atol=1e-4)
    assert torch.allclose(m1[0], layer.mlp(h[0]), atol=1e-4)


class MockedDTP(DTPQwen3):
    """Replaces module computation with recorded x-independent tensors so the
    residual/queue bookkeeping can be checked against the blog equations."""

    def _module_out(self, layer, kind, x, **kw):
        n = len(self.recorded)
        g = torch.Generator().manual_seed(1000 + n)
        o = torch.randn(self.L, B, S, self.config.hidden_size, generator=g)
        self.recorded.append(o)
        return o


@pytest.mark.parametrize("delta", [0, 0.5, 1, 1.5, 2.5, 3, 3.75, 4])
def test_orchestrator_matches_blog_equations(delta):
    hf = tiny_model()
    L = 4
    dtp = MockedDTP(hf, n_devices=L, delta=delta)
    dtp.recorded = []
    x_ids = ids()
    got = dtp(x_ids).logits
    outs = dtp.recorded
    n_modules = dtp.n_modules
    assert len(outs) == n_modules

    # Literal per-device reference of the three-stage equations, sender-side:
    # the broadcast of module m lands (1 - frac) at m + d and frac at m + d + 1.
    d_int, frac = int(delta), delta - int(delta)
    emb = hf.model.embed_tokens(x_ids)
    streams = []
    for l in range(L):
        X = emb.clone()
        for n in range(n_modules):
            X = X + dtp._own_scale(n) * outs[n][l]
            for m in range(n_modules):
                for arrive, w in ((m + d_int, 1.0 - frac), (m + d_int + 1, frac)):
                    if arrive == n and w > 0:
                        for j in range(L):
                            if j != l:
                                X = X + w * outs[m][j]
        streams.append(hf.model.norm(X))
    ref = hf.lm_head(torch.stack(streams).mean(0))
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


def test_fractional_delta_is_continuous():
    hf = tiny_model()
    x = ids()
    dtp = DTPQwen3(hf, n_devices=4, delta=0)

    def logits(dv):
        dtp.delta = dv
        with torch.no_grad():
            return dtp(x).logits

    big = (logits(0.0) - logits(4.0)).abs().max().item()
    assert big > 1e-3  # delta genuinely changes the function on this model
    eps = 1e-3
    for base in (0.0, 0.999, 1.0, 2.5, 3.999):  # crosses integer boundaries
        step = (logits(base) - logits(base + eps)).abs().max().item()
        assert step < 0.02 * big + 1e-5, (base, step, big)


@pytest.mark.parametrize("delta", [2, 1.5])
def test_kv_cache_matches_full_forward(delta):
    hf = tiny_model()
    dtp = DTPQwen3(hf, n_devices=2, delta=delta)  # hq=4, kvp=2 -> exercises GQA expand
    x = ids(1, 12)
    full = dtp(x).logits

    cache = DTPCache()
    out = dtp(x[:, :8], cache=cache)
    assert torch.allclose(out.logits, full[:, :8], atol=1e-4)
    for t in range(8, 12):
        out = dtp(x[:, t : t + 1], cache=cache)
        assert torch.allclose(out.logits[:, 0], full[:, t], atol=1e-4), t


def test_gradient_checkpointing_matches_and_flows():
    hf = tiny_model()
    x = ids()
    dtp = DTPQwen3(hf, n_devices=4, delta=3)
    dtp.train()
    loss_ref = dtp(x, labels=x).loss
    loss_ref.backward()
    ref_grad = hf.model.layers[2].self_attn.q_proj.weight.grad.clone()
    hf.zero_grad(set_to_none=True)

    dtp.gradient_checkpointing = True
    loss_ckpt = dtp(x, labels=x).loss
    assert torch.allclose(loss_ref, loss_ckpt, atol=1e-5)
    loss_ckpt.backward()
    got_grad = hf.model.layers[2].self_attn.q_proj.weight.grad
    assert got_grad is not None and got_grad.abs().sum() > 0
    assert torch.allclose(ref_grad, got_grad, atol=1e-5)


def test_delta_bounds():
    hf = tiny_model()
    with pytest.raises(AssertionError):
        DTPQwen3(hf, n_devices=2, delta=5)  # > n_modules // 2 = 4
    with pytest.raises(AssertionError):
        DTPQwen3(hf, n_devices=2, delta=4.5)
    DTPQwen3(hf, n_devices=2, delta=4)


def test_delta_scheduler():
    from scripts.finetune import DeltaScheduler

    s = DeltaScheduler(slope=0.005, ramp=100, cap=28, gate_ppl=20.0)
    for _ in range(50):
        v = s.step()
    assert v == pytest.approx(0.000025 * 50**2)  # quadratic ramp
    for _ in range(50):
        v = s.step()
    assert v == pytest.approx(0.25)  # C1 joint: 0.005*100 - 0.25
    for _ in range(750):
        v = s.step()
    assert v == pytest.approx(4.0)  # crosses delta=4 at s=850
    s.gate(25.0)  # ppl above the gate: clock freezes
    for _ in range(100):
        v = s.step()
    assert v == pytest.approx(4.0)
    s.gate(15.0)  # back under: resumes where it left off
    for _ in range(1150):
        v = s.step()
    assert v == pytest.approx(9.75)  # nominal end-of-run value at s=2000

    s2 = DeltaScheduler(slope=0.005, ramp=100, cap=1.0, gate_ppl=20.0)
    for _ in range(2000):
        v2 = s2.step()
    assert v2 == 1.0


def test_generate_runs():
    hf = tiny_model()
    dtp = DTPQwen3(hf, n_devices=4, delta=4)
    out = dtp.generate(ids(1, 5), max_new_tokens=4)
    assert out.shape == (1, 9)


def test_llama_l1_delta0_matches_hf():
    """The wrapper is not Qwen3-specific: a Llama block (no q/k norm, tied
    head_dim = hidden / heads) must match too."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=4, max_position_embeddings=256,
    )
    torch.manual_seed(7)
    hf = LlamaForCausalLM(cfg).eval()
    x = ids()
    ref = hf(x).logits
    for L in (1, 4):
        got = DTPQwen3(hf, n_devices=L, delta=0)(x).logits
        assert torch.allclose(ref, got, atol=2e-4, rtol=1e-4), (L, (ref - got).abs().max())
