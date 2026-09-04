"""Collect head<->neuron affinity statistics for "expertised" DTP sharding
(delta = 1: each module only misses the other devices' output of the
immediately preceding module).

Two edges per layer i:
  up:   attention heads of layer i      -> FFN neurons of layer i
  down: FFN neurons of layer i          -> KV groups (q/k/v) of layer i+1

For each edge, three scores over calibration tokens (all mean-squared over tokens):
  add : the additive piece of the source's contribution to the target's
        pre-activation, through the RMSNorm gain, divided by the *vanilla* rms
        of the token (exact decomposition of the vanilla pre-activation).
  abl : leave-one-out ablation of the pre-activation: pre(full) - pre(full - source),
        each with its own rms (so it includes the per-token rescaling artifact).
  fo  : (up edge only) first-order propagation of the additive piece through
        SwiGLU at the vanilla operating point, weighted by ||down column||^2,
        i.e. damage in residual-stream units.

Saves per-layer matrices and prints Spearman rank correlations add-vs-abl.
"""

import argparse
import sys
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import batched, packed_stream


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--blocks", type=int, default=64, help="calibration blocks of seq-len tokens")
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--out", default="runs/affinity_stats.pt")
    p.add_argument("--no-abl", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model_id)
    hf = Qwen3ForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()
    cfg = hf.config
    layers = hf.model.layers
    NL, D, H, KV, I = cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size
    hd = layers[0].self_attn.head_dim
    hpg = H // KV  # q heads per KV group
    eps = cfg.rms_norm_eps
    print(f"L={NL} D={D} H={H} KV={KV} hd={hd} I={I}")

    # ---- hooks: capture per-layer activations for one micro-batch
    cap = {}

    def pre_hook(name):
        def f(mod, inp):
            cap[name] = inp[0].detach()
        return f

    def out_hook(name):
        def f(mod, inp, out):
            cap[name] = out.detach()
        return f

    for i, layer in enumerate(layers):
        layer.self_attn.o_proj.register_forward_pre_hook(pre_hook((i, "a")))  # [B,S,H*hd]
        layer.post_attention_layernorm.register_forward_pre_hook(pre_hook((i, "x_mlp")))  # residual before MLP norm
        layer.input_layernorm.register_forward_pre_hook(pre_hook((i, "x_attn")))  # residual before attn norm
        layer.mlp.gate_proj.register_forward_hook(out_hook((i, "g")))
        layer.mlp.up_proj.register_forward_hook(out_hook((i, "u")))
        layer.mlp.down_proj.register_forward_pre_hook(pre_hook((i, "act")))

    # ---- accumulators
    Z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float64)
    S_up_add, S_up_abl, S_up_fo = Z(NL, H, I), Z(NL, H, I), Z(NL, H, I)
    S_dn_add, S_dn_abl = Z(NL, I, KV), Z(NL, I, KV)
    rms_shift = Z(NL, H)  # mean |rms(x - head) / rms(x) - 1|
    pre_full_sq = Z(NL, I)  # mean pre-activation^2 (gate+up), for scale reference
    n_tok = 0

    # per-layer static pieces for the down edge
    Wqkv = []  # [KV, hpg*hd + 2hd, D]
    for i, layer in enumerate(layers):
        at = layer.self_attn
        Wq = at.q_proj.weight.view(KV, hpg * hd, D)
        Wk = at.k_proj.weight.view(KV, hd, D)
        Wv = at.v_proj.weight.view(KV, hd, D)
        Wqkv.append(torch.cat([Wq, Wk, Wv], dim=1))

    def silu_d(g):
        s = torch.sigmoid(g)
        return s * (1 + g * (1 - s))

    stream = packed_stream(tok, args.seq_len, args.dataset, args.dataset_config)
    with torch.no_grad():
        for bi, blk in enumerate(islice(batched(stream, args.micro_batch), args.blocks // args.micro_batch)):
            blk = blk.to(device)
            cap.clear()
            hf(blk[:, :-1])
            T = blk.shape[0] * (blk.shape[1] - 1)
            n_tok += T
            for i, layer in enumerate(layers):
                # ---------------- up edge: heads(i) -> neurons(i)
                a = cap[(i, "a")].reshape(T, H, hd)
                Wo = layer.self_attn.o_proj.weight.view(D, H, hd)
                head_out = torch.einsum("the,dhe->htd", a, Wo)  # [H,T,D]
                x = cap[(i, "x_mlp")].reshape(T, D)
                gain = layer.post_attention_layernorm.weight
                rms = (x.pow(2).mean(-1) + eps).sqrt()  # [T]
                Wg, Wu = layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight  # [I,D]
                piece = head_out * gain / rms[None, :, None]
                pg = piece @ Wg.T  # [H,T,I]
                pu = piece @ Wu.T
                S_up_add[i] += (pg.pow(2) + pu.pow(2)).sum(1).double()
                g, u = cap[(i, "g")].reshape(T, I), cap[(i, "u")].reshape(T, I)
                pre_full_sq[i] += (g.pow(2) + u.pow(2)).sum(0).double()
                wd2 = layer.mlp.down_proj.weight.pow(2).sum(0)  # [I]
                dout = silu_d(g)[None] * u[None] * pg + F.silu(g)[None] * pu
                S_up_fo[i] += (dout.pow(2).sum(1) * wd2).double()
                del dout
                if not args.no_abl:
                    x_wo = x[None] - head_out  # [H,T,D]
                    rms_wo = (x_wo.pow(2).mean(-1) + eps).sqrt()  # [H,T]
                    rms_shift[i] += (rms_wo / rms[None] - 1).abs().sum(1).double()
                    full_g = ((x * gain) / rms[:, None]) @ Wg.T  # [T,I]
                    full_u = ((x * gain) / rms[:, None]) @ Wu.T
                    normed_wo = x_wo * gain / rms_wo[..., None]
                    d_g = full_g[None] - normed_wo @ Wg.T
                    d_u = full_u[None] - normed_wo @ Wu.T
                    S_up_abl[i] += (d_g.pow(2) + d_u.pow(2)).sum(1).double()
                    del x_wo, normed_wo, d_g, d_u
                del pg, pu, piece, head_out

                # ---------------- down edge: neurons(i) -> KV groups(i+1)
                if i + 1 < NL:
                    nxt = layers[i + 1]
                    act = cap[(i, "act")].reshape(T, I)
                    xn = cap[(i + 1, "x_attn")].reshape(T, D)
                    gain_n = nxt.input_layernorm.weight
                    rms_n = (xn.pow(2).mean(-1) + eps).sqrt()  # [T]
                    Wd = layer.mlp.down_proj.weight  # [D,I]
                    Bm = Wqkv[i + 1] @ (Wd * gain_n[:, None])  # [KV, R, I]
                    B2 = Bm.pow(2).sum(1)  # [KV, I]
                    E = (act.pow(2) / rms_n[:, None].pow(2)).sum(0)  # [I]
                    S_dn_add[i] += (B2.T * E[:, None]).double()
                    if not args.no_abl:
                        A = torch.einsum("grd,td->gtr", Wqkv[i + 1], xn * gain_n)  # [KV,T,R]
                        A2 = A.pow(2).sum(-1)  # [KV,T]
                        AB = torch.bmm(A, Bm)  # [KV,T,I]
                        xw = xn @ Wd  # [T,I] = x'.Wd_j
                        wd2 = Wd.pow(2).sum(0)  # [I]
                        xn2 = xn.pow(2).sum(-1)  # [T]
                        rj2 = (xn2[:, None] - 2 * act * xw + act.pow(2) * wd2[None]) / D + eps
                        inv_rj = rj2.clamp_min(1e-12).rsqrt()  # [T,I]
                        inv_r = 1 / rms_n  # [T]
                        c = (inv_r[:, None] - inv_rj)  # [T,I]
                        s = act * inv_rj  # [T,I]
                        diff2 = c.pow(2)[None] * A2[..., None] + 2 * c[None] * s[None] * AB + s.pow(2)[None] * B2[:, None, :]
                        S_dn_abl[i] += diff2.sum(1).T.double()
                        del A, AB, diff2
            if bi % 4 == 0:
                print(f"batch {bi}: {n_tok} tokens", flush=True)

    out = dict(
        n_tok=n_tok, H=H, KV=KV, I=I, NL=NL, hpg=hpg,
        S_up_add=(S_up_add / n_tok).cpu(), S_up_abl=(S_up_abl / n_tok).cpu(), S_up_fo=(S_up_fo / n_tok).cpu(),
        S_dn_add=(S_dn_add / n_tok).cpu(), S_dn_abl=(S_dn_abl / n_tok).cpu(),
        rms_shift=(rms_shift / n_tok).cpu(), pre_full_sq=(pre_full_sq / n_tok).cpu(),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {args.out} ({n_tok} tokens)")
    report(out)


def concentration(M, L):
    """Share of each target's affinity mass on its top source, vs 1/n_sources,
    plus expected co-located fraction of a random balanced partition (1/L)."""
    tot = M.sum(0, keepdim=True).clamp_min(1e-30)
    frac = M / tot
    return frac.max(0).values.mean().item()


def report(o):
    from scipy.stats import spearmanr

    NL, H, KV, hpg = o["NL"], o["H"], o["KV"], o["hpg"]
    print("\nrms shift when removing one head (mean |rms_wo/rms - 1|), per layer:")
    print(" ".join(f"{v:.3f}" for v in o["rms_shift"].mean(1).tolist()))
    print("\nlayer | spearman up add~abl | up add~fo | up abl~fo | down add~abl | top-KVgroup share (up, add) | top-KVgroup share (down, add)")
    for i in range(NL):
        up_add = o["S_up_add"][i].view(KV, hpg, -1).sum(1)  # per KV group
        up_abl = o["S_up_abl"][i].view(KV, hpg, -1).sum(1)
        up_fo = o["S_up_fo"][i].view(KV, hpg, -1).sum(1)
        r1 = spearmanr(up_add.flatten(), up_abl.flatten()).correlation
        r2 = spearmanr(up_add.flatten(), up_fo.flatten()).correlation
        r3 = spearmanr(up_abl.flatten(), up_fo.flatten()).correlation
        if i + 1 < NL:
            r4 = spearmanr(o["S_dn_add"][i].flatten(), o["S_dn_abl"][i].flatten()).correlation
            c_dn = concentration(o["S_dn_add"][i].T, 4)
        else:
            r4, c_dn = float("nan"), float("nan")
        c_up = concentration(up_add, 4)
        print(f"{i:5d} | {r1:8.3f}            | {r2:8.3f}  | {r3:8.3f}  | {r4:8.3f}     | {c_up:.3f} (uniform {1/KV:.3f})          | {c_dn:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(torch.load(sys.argv[2]))
    else:
        main()
