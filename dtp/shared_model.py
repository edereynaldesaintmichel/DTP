"""DTP with a shared expert: a slice of every layer's FFN neurons is replicated
on all virtual devices instead of being sharded.

Layout convention (per layer, after the layout permutation has been applied to
the HF weights with scripts/expertise.apply_permutation):

    neurons [0, I - n_shared)   -> sharded, contiguous shard per device
    neurons [I - n_shared, I)   -> shared, computed in full on every device

Each device adds the full output of the shared neurons to its own residual
immediately and never broadcasts it. The broadcast partial is computed from the
device's shard only, so the shared neurons are counted exactly once:

    residual = sum over devices of the (delayed) shard partials
             + the shared piece, computed locally from this device's residual

With no delay every device's residual is identical and this equals the vanilla
model. Only the FFN has a shared part; attention is sharded as in DTPQwen3
(a shared KV group would need uneven head shards).

In a permutation file (*.perm.pt), a shared neuron is marked with device id L
(one past the last real device): stable argsort then puts it last.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .model import DTPOutput, DTPQwen3


def n_shared_for(intermediate_size, n_devices, shared_frac):
    """Smallest neuron count >= shared_frac * I that leaves the rest divisible
    by n_devices. 10% of 3072 on 4 devices -> 308 (2764 = 4 * 691 sharded)."""
    n = int(round(shared_frac * intermediate_size))
    while (intermediate_size - n) % n_devices:
        n += 1
    return n


class SharedDTPQwen3(DTPQwen3):
    def __init__(self, hf_model, n_devices=4, delta=4, n_shared=0, **kw):
        I = hf_model.config.intermediate_size
        assert 0 <= n_shared < I and (I - n_shared) % n_devices == 0, (
            f"I - n_shared = {I - n_shared} must be divisible by n_devices = {n_devices}"
        )
        super().__init__(hf_model, n_devices=n_devices, delta=delta, **kw)
        self.n_shared = n_shared

    # ----------------------------------------------------------------- shards

    def _mlp_partials(self, layer, x):
        """x: [L, B, S, D] (already normed). Returns (shard_partial, shared_piece),
        both [L, B, S, D]. shard partials sum to the sharded neurons' output;
        shared_piece is the full output of the shared neurons on each device."""
        mlp = layer.mlp
        Lc, _, _, D = x.shape
        I = self.config.intermediate_size
        n_sharded = I - self.n_shared
        Ip = n_sharded // Lc
        Wg_all, Wu_all, Wd_all = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight

        Wg = Wg_all[:n_sharded].view(Lc, Ip, D)
        Wu = Wu_all[:n_sharded].view(Lc, Ip, D)
        Wd = Wd_all[:, :n_sharded].view(D, Lc, Ip)
        g = torch.einsum("lbsd,lid->lbsi", x, Wg)
        u = torch.einsum("lbsd,lid->lbsi", x, Wu)
        partial = torch.einsum("lbsi,dli->lbsd", mlp.act_fn(g) * u, Wd)
        if self.n_shared == 0:
            return partial, None

        Wg_sh, Wu_sh, Wd_sh = Wg_all[n_sharded:], Wu_all[n_sharded:], Wd_all[:, n_sharded:]
        g_sh = x @ Wg_sh.T  # [L, B, S, n_shared], one copy per device
        u_sh = x @ Wu_sh.T
        shared = (mlp.act_fn(g_sh) * u_sh) @ Wd_sh.T
        return partial, shared

    def _module_out(self, layer, kind, x, pos_emb=None, mask=None, is_causal=False, cache=None):
        """Returns (broadcast_partial, local_piece). local_piece is None for attention."""
        if kind == "attn":
            h = layer.input_layernorm(x)
            return self._attn_partials(layer, h, pos_emb, mask, is_causal, cache), None
        h = layer.post_attention_layernorm(x)
        return self._mlp_partials(layer, h)

    # ---------------------------------------------------------------- forward

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
        queue = []  # in-flight (module_idx, broadcast_partial) pairs
        n = 0
        for layer in model.layers:
            for kind in ("attn", "mlp"):
                kw = dict(pos_emb=pos_emb, mask=mask, is_causal=is_causal, cache=cache) if kind == "attn" else {}
                if use_ckpt:
                    o, local = checkpoint(
                        lambda t, _layer=layer, _kind=kind, _kw=kw: self._module_out(_layer, _kind, t, **_kw),
                        x,
                        use_reentrant=False,
                    )
                else:
                    o, local = self._module_out(layer, kind, x, **kw)
                x = x + self._own_scale(n) * o
                if local is not None:
                    x = x + local  # complete on this device: no scale, no broadcast
                queue.append((n, o))
                if self.L > 1:
                    for m, w in ((n - d, 1.0 - a), (n - d - 1, a)):
                        if w > 0.0 and m >= 0:
                            o_past = queue[m - queue[0][0]][1]
                            x = x + w * (o_past.sum(dim=0, keepdim=True) - o_past)
                while queue and queue[0][0] <= n - d - 1:
                    queue.pop(0)
                n += 1

        x = model.norm(x)
        x = x.mean(dim=0)
        logits = self.hf.lm_head(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return DTPOutput(logits=logits, loss=loss)


def mark_shared(neur_dev, shared_idx, L):
    """neur_dev: [I] device ids of the sharded neurons (values for shared_idx are
    ignored). Returns a copy with the shared neurons set to the sentinel L."""
    out = neur_dev.clone()
    out[shared_idx] = L
    return out


def check_layout(neur_dev, L, n_shared):
    """Every device holds (I - n_shared) / L neurons and exactly n_shared are shared."""
    I = neur_dev.numel()
    counts = torch.bincount(neur_dev, minlength=L + 1)
    assert counts[L] == n_shared, f"{counts[L]} shared neurons, expected {n_shared}"
    assert (counts[:L] == (I - n_shared) // L).all(), f"unbalanced shards: {counts[:L].tolist()}"
