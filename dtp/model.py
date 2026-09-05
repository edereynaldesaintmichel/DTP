"""Delayed Tensor Parallelism (DTP) for Qwen3, after
https://blog.kog.ai/delayed-tensor-parallelism-for-faster-transformer-inference/

The model is sharded per-module (attention by query heads, MLP by intermediate
columns) across L *virtual* devices simulated on one GPU. Each device keeps its
own residual stream X_l. With module index n (2 modules per layer: attention,
MLP) and delay delta:

  stage 1 (n < delta):            X_l <- X_l + sqrt(L) * o_l^n
  stage 2 (delta <= n < 2N-delta): X_l <- X_l + o_l^n + sum_{j!=l} o_j^{n-delta}
  stage 3 (n >= 2N-delta):        own output added (sqrt(L)-scaled by default);
                                  in-flight aggregations from n-delta still land,
                                  but no new broadcast is initiated.

At the end, the final RMSNorm is applied per device and the L outputs are
averaged (the blog's final all-reduce; averaging after the linear LM head is
identical to averaging before it, so the head runs once).

delta may be fractional: with delta = d + a (0 < a < 1), the broadcast of
module m lands with weight (1 - a) at module m + d and weight a at module
m + d + 1, and the stage-1/3 sqrt(L) own-scales are interpolated the same way
— at each step the update is the convex combination of the delta = d and
delta = d + 1 dynamics, so the forward is continuous in delta. `delta` is a
plain attribute and may be changed between forwards (e.g. annealed during
training).

Qwen3 is a standard pre-norm architecture: each module's output is added to the
residual raw, with no norm inside the branch. The per-device partial outputs
therefore sum exactly to the vanilla module output and the blog's equations
apply verbatim — no norm adaptation needed. (This is why Qwen3 was chosen over
Gemma 3, whose sandwich post-norms act on the full module output that no device
holds under DTP.)

All shard weights are views into the wrapped HF model's parameters: training the
DTP wrapper updates the HF model in place, and checkpoints remain valid HF
state dicts.

Any pre-norm GQA/SwiGLU model with the HF Llama module layout works (Qwen3,
Llama, h2o-danube3, ...): the q/k head norms are applied only if present.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(t, cos, sin):
    # t: [L, B, h, S, e]; cos/sin: [B, S, e]
    cos = cos[None, :, None]
    sin = sin[None, :, None]
    return t * cos + _rotate_half(t) * sin


class DTPCache:
    """Per-(virtual-device, layer) KV cache. K/V are stored with the L dim
    folded into the batch dim: [L*B, kv_heads, S, head_dim]."""

    def __init__(self):
        self.kv = {}

    @property
    def seq_len(self):
        if not self.kv:
            return 0
        k, _ = next(iter(self.kv.values()))
        return k.shape[2]

    def update(self, layer_idx, k, v):
        if layer_idx in self.kv:
            pk, pv = self.kv[layer_idx]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        self.kv[layer_idx] = (k, v)
        return k, v


@dataclass
class DTPOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class DTPQwen3(nn.Module):
    def __init__(
        self,
        hf_model,
        n_devices: int = 4,
        delta: float = 4,
        stage3_own_scale: str = "sqrt_l",  # "sqrt_l" | "one"
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        cfg = hf_model.config
        L = n_devices
        self.n_modules = 2 * cfg.num_hidden_layers
        assert cfg.num_attention_heads % L == 0, "n_devices must divide num_attention_heads"
        assert cfg.num_key_value_heads % L == 0, "n_devices must divide num_key_value_heads"
        assert cfg.intermediate_size % L == 0, "n_devices must divide intermediate_size"
        assert not cfg.attention_bias, "biased attention projections not supported"
        assert not getattr(cfg, "use_sliding_window", False) and getattr(cfg, "sliding_window", None) is None
        assert 0 <= delta <= self.n_modules // 2, (
            f"delta must be in [0, {self.n_modules // 2}] (it is counted in modules, 2 per layer)"
        )
        assert stage3_own_scale in ("sqrt_l", "one")

        self.hf = hf_model
        self.L = L
        self.delta = delta
        self.stage3_own_scale = stage3_own_scale
        self.gradient_checkpointing = gradient_checkpointing

    @property
    def config(self):
        return self.hf.config

    # ------------------------------------------------------------------ shards

    def _attn_partials(self, layer, x, pos_emb, mask, is_causal, cache=None):
        """x: [L, B, S, D] (already input-layernormed). Returns per-device raw
        partial attention outputs [L, B, S, D]; their sum over devices equals the
        full attention output when all device streams are identical."""
        attn = layer.self_attn
        cfg = self.config
        Lc, B, S, D = x.shape
        hd = attn.head_dim  # Qwen3 has an explicit head_dim with num_heads * head_dim != hidden_size
        H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
        hq = H // Lc
        kvp = KV // Lc

        Wq = attn.q_proj.weight.view(Lc, hq, hd, D)
        Wk = attn.k_proj.weight.view(Lc, kvp, hd, D)
        Wv = attn.v_proj.weight.view(Lc, kvp, hd, D)
        q = torch.einsum("lbsd,lhed->lbshe", x, Wq)
        k = torch.einsum("lbsd,lhed->lbshe", x, Wk)
        v = torch.einsum("lbsd,lhed->lbshe", x, Wv)

        if hasattr(attn, "q_norm"):  # Qwen3: RMSNorm over head_dim, before RoPE (Llama has none)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q = q.permute(0, 1, 3, 2, 4)  # [L, B, h, S, e]
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        cos, sin = pos_emb
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        q = q.reshape(Lc * B, hq, S, hd)
        k = k.reshape(Lc * B, kvp, S, hd)
        v = v.reshape(Lc * B, kvp, S, hd)
        if cache is not None:
            k, v = cache.update(attn.layer_idx, k, v)
        rep = hq // kvp
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, scale=attn.scaling
        )
        out = out.view(Lc, B, hq, S, hd).permute(0, 1, 3, 2, 4)  # [L, B, S, h, e]
        Wo = attn.o_proj.weight.view(D, Lc, hq, hd)
        return torch.einsum("lbshe,dlhe->lbsd", out, Wo)

    def _mlp_partials(self, layer, x):
        """x: [L, B, S, D] (already post-attention-layernormed)."""
        mlp = layer.mlp
        cfg = self.config
        Lc, _, _, D = x.shape
        Ip = cfg.intermediate_size // Lc
        Wg = mlp.gate_proj.weight.view(Lc, Ip, D)
        Wu = mlp.up_proj.weight.view(Lc, Ip, D)
        Wd = mlp.down_proj.weight.view(D, Lc, Ip)
        g = torch.einsum("lbsd,lid->lbsi", x, Wg)
        u = torch.einsum("lbsd,lid->lbsi", x, Wu)
        return torch.einsum("lbsi,dli->lbsd", mlp.act_fn(g) * u, Wd)

    def _module_out(self, layer, kind, x, pos_emb=None, mask=None, is_causal=False, cache=None):
        """Per-device partial output of one module. [L, B, S, D]. Pre-norm: the
        partial is raw, and partials sum exactly to the vanilla module output."""
        if kind == "attn":
            h = layer.input_layernorm(x)
            return self._attn_partials(layer, h, pos_emb, mask, is_causal, cache)
        h = layer.post_attention_layernorm(x)  # despite the name: the MLP's pre-norm
        return self._mlp_partials(layer, h)

    # ----------------------------------------------------------------- forward

    def _own_scale_at(self, n, delta):
        if delta == 0:
            return 1.0
        if n < delta:
            return math.sqrt(self.L)
        if n >= self.n_modules - delta and self.stage3_own_scale == "sqrt_l":
            return math.sqrt(self.L)
        return 1.0

    def _own_scale(self, n):
        d = int(self.delta)
        a = self.delta - d
        if a == 0:
            return self._own_scale_at(n, d)
        return (1 - a) * self._own_scale_at(n, d) + a * self._own_scale_at(n, d + 1)

    @staticmethod
    def _build_mask(S, past, device):
        """Returns (attn_mask, is_causal) for SDPA. Bool mask: True = attend."""
        if past == 0 and S > 1:
            return None, True
        if S == 1:
            return None, False
        i = torch.arange(S, device=device)[:, None] + past
        j = torch.arange(past + S, device=device)[None, :]
        return (j <= i)[None, None], False

    def forward(self, input_ids, labels=None, cache=None):
        model = self.hf.model
        B, S = input_ids.shape
        device = input_ids.device
        past = cache.seq_len if cache is not None else 0
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        assert not (use_ckpt and cache is not None)

        h = model.embed_tokens(input_ids)
        position_ids = torch.arange(past, past + S, device=device)[None].expand(B, S)
        pos_emb = model.rotary_emb(h, position_ids)
        mask, is_causal = self._build_mask(S, past, device)

        x = h.unsqueeze(0).expand(self.L, B, S, h.shape[-1])
        d = int(self.delta)
        a = self.delta - d
        queue = []  # in-flight (module_idx, partial_output) pairs
        n = 0
        for layer in model.layers:
            for kind in ("attn", "mlp"):
                kw = dict(pos_emb=pos_emb, mask=mask, is_causal=is_causal, cache=cache) if kind == "attn" else {}
                if use_ckpt:
                    o = checkpoint(
                        lambda t, _layer=layer, _kind=kind, _kw=kw: self._module_out(_layer, _kind, t, **_kw),
                        x,
                        use_reentrant=False,
                    )
                else:
                    o = self._module_out(layer, kind, x, **kw)
                x = x + self._own_scale(n) * o
                queue.append((n, o))
                if self.L > 1:
                    # broadcasts from module m land (1 - a) at m + d, a at m + d + 1
                    for m, w in ((n - d, 1.0 - a), (n - d - 1, a)):
                        if w > 0.0 and m >= 0:
                            o_past = queue[m - queue[0][0]][1]
                            x = x + w * (o_past.sum(dim=0, keepdim=True) - o_past)
                while queue and queue[0][0] <= n - d - 1:
                    queue.pop(0)
                n += 1
        # entries still in `queue` are the trailing modules whose cross-device
        # broadcasts never (fully) land before the end of the network (stage 3).

        x = model.norm(x)  # per-device final norm
        x = x.mean(dim=0)  # final all-reduce average; LM head is linear so
        logits = self.hf.lm_head(x)  # averaging before the head is equivalent

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return DTPOutput(logits=logits, loss=loss)

    # ---------------------------------------------------------------- generate

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=0.0, top_k=None, eos_token_id=None):
        self.eval()
        cache = DTPCache()
        out = self(input_ids, cache=cache)
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = out.logits[:, -1].float()
            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                next_id = torch.multinomial(logits.softmax(-1), 1)
            else:
                next_id = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
            out = self(next_id, cache=cache)
        return ids


def load_dtp_qwen3(
    model_id="Qwen/Qwen3-0.6B-Base",
    n_devices=4,
    delta=4,
    dtype=torch.bfloat16,
    device="cuda",
    state_dict_path=None,
    stage3_own_scale="sqrt_l",
    gradient_checkpointing=False,
):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if state_dict_path:
        sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        hf.load_state_dict(sd)
    hf.to(device)
    return DTPQwen3(
        hf,
        n_devices=n_devices,
        delta=delta,
        stage3_own_scale=stage3_own_scale,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)
