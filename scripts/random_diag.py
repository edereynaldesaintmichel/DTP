"""Why do random balanced permutations differ so much in untrained DTP loss?

1. Decompose: heads-only vs neurons-only permutation for the best/worst seeds.
2. Trace: per-module relative error of each device's residual vs the vanilla
   residual (and rms ratio), to see where the streams diverge.
3. Layer swap: start from the worst seed, substitute the best seed's layer-k
   permutation one layer at a time, and see which layers move the loss.
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import perplexity, wikitext_blocks
from dtp.model import DTPQwen3
from scripts.expertise import apply_permutation, random_partition


@torch.no_grad()
def trace(dtp, ids):
    """Returns list over modules of (rel_err [L], rms_ratio [L], max_abs_dtp, max_abs_vanilla)."""
    hf, model = dtp.hf, dtp.hf.model
    van = {}
    hooks = []
    for i, layer in enumerate(model.layers):
        hooks.append(layer.input_layernorm.register_forward_pre_hook(lambda m, inp, i=i: van.__setitem__(2 * i - 1, inp[0])))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, inp, i=i: van.__setitem__(2 * i, inp[0])))
    hooks.append(model.norm.register_forward_pre_hook(lambda m, inp: van.__setitem__(2 * len(model.layers) - 1, inp[0])))
    hf(ids)
    for h in hooks:
        h.remove()

    # replicate DTPQwen3.forward with snapshots
    B, S = ids.shape
    h = model.embed_tokens(ids)
    pos = torch.arange(S, device=ids.device)[None].expand(B, S)
    pos_emb = model.rotary_emb(h, pos)
    x = h.unsqueeze(0).expand(dtp.L, B, S, h.shape[-1])
    d, a = int(dtp.delta), dtp.delta - int(dtp.delta)
    queue, n, out = [], 0, []
    for layer in model.layers:
        for kind in ("attn", "mlp"):
            kw = dict(pos_emb=pos_emb, mask=None, is_causal=True) if kind == "attn" else {}
            o = dtp._module_out(layer, kind, x, **kw)
            x = x + dtp._own_scale(n) * o
            queue.append((n, o))
            for m, w in ((n - d, 1.0 - a), (n - d - 1, a)):
                if w > 0.0 and m >= 0:
                    op = queue[m - queue[0][0]][1]
                    x = x + w * (op.sum(0, keepdim=True) - op)
            while queue and queue[0][0] <= n - d - 1:
                queue.pop(0)
            v = van[n]  # vanilla residual after module n
            err = (x - v[None]).flatten(1).norm(dim=1) / v.norm()
            rms = x.pow(2).mean(-1).sqrt().mean((1, 2)) / v.pow(2).mean(-1).sqrt().mean()
            out.append((err.cpu(), rms.cpu(), x.abs().max().item(), v.abs().max().item()))
            n += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--delta", type=float, default=1)
    p.add_argument("--good-seed", type=int, default=3)
    p.add_argument("--bad-seed", type=int, default=0)
    p.add_argument("--n-blocks", type=int, default=32)
    p.add_argument("--swap-blocks", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    args = p.parse_args()

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    device, L = "cuda", args.devices
    tok = AutoTokenizer.from_pretrained(args.model_id)
    blocks = wikitext_blocks(tok, args.seq_len, args.n_blocks)
    base_sd = Qwen3ForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).state_dict()
    hf = Qwen3ForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()
    dtp = DTPQwen3(hf, n_devices=L, delta=args.delta).to(device).eval()
    cfg = hf.config
    KV, I, NL = cfg.num_key_value_heads, cfg.intermediate_size, cfg.num_hidden_layers

    def parts_for(seed):
        gen = torch.Generator().manual_seed(1000 + seed)
        ps = [random_partition(KV, I, L, gen) for _ in range(NL)]
        return [h for h, _ in ps], [n for _, n in ps]

    ident_h = [torch.arange(KV) // (KV // L) for _ in range(NL)]
    ident_n = [torch.arange(I) // (I // L) for _ in range(NL)]

    def ev(h, n, blk=blocks):
        hf.load_state_dict(base_sd)
        apply_permutation(hf, h, n, L)
        _, nll = perplexity(lambda x: dtp(x).logits, blk, device, 4)
        return nll

    good_h, good_n = parts_for(args.good_seed)
    bad_h, bad_n = parts_for(args.bad_seed)
    print(f"delta={args.delta} L={L}  nll on {args.n_blocks} wikitext blocks")
    print(f"  identity            {ev(ident_h, ident_n):.4f}")
    for name, (h, n) in (("good", (good_h, good_n)), ("bad", (bad_h, bad_n))):
        print(f"  {name} full           {ev(h, n):.4f}")
        print(f"  {name} heads only     {ev(h, ident_n):.4f}")
        print(f"  {name} neurons only   {ev(ident_h, n):.4f}")
    print(f"  good heads + bad neurons  {ev(good_h, bad_n):.4f}")
    print(f"  bad heads + good neurons  {ev(bad_h, good_n):.4f}", flush=True)

    # ---- trace
    ids = blocks[:2, :-1].to(device)
    print("\nper-module relative error of device residuals vs vanilla (mean over devices), rms ratio, max|x| dtp / vanilla")
    print("module | identity            | good                | bad")
    traces = {}
    for name, (h, n) in (("identity", (ident_h, ident_n)), ("good", (good_h, good_n)), ("bad", (bad_h, bad_n))):
        hf.load_state_dict(base_sd)
        apply_permutation(hf, h, n, L)
        traces[name] = trace(dtp, ids)
    for m in range(2 * NL):
        row = f"{m:4d} {'attn' if m % 2 == 0 else 'mlp '} |"
        for name in ("identity", "good", "bad"):
            err, rms, mx, mv = traces[name][m]
            row += f" err {err.mean():.3f} rms {rms.mean():.2f} max {mx:8.1f}/{mv:8.1f} |"
        print(row)
    print("\nper-device relative error at the last module:")
    for name in ("identity", "good", "bad"):
        print(f"  {name}: " + " ".join(f"{v:.3f}" for v in traces[name][-1][0].tolist()))

    # ---- layer swap: bad base, substitute good layer k
    blk = blocks[: args.swap_blocks]
    base = ev(bad_h, bad_n, blk)
    print(f"\nlayer swap (bad base nll {base:.4f} on {args.swap_blocks} blocks): substitute good seed's layer k")
    print("layer | heads+neurons | heads only | neurons only")
    for k in range(NL):
        h = list(bad_h); n = list(bad_n)
        h[k], n[k] = good_h[k], good_n[k]
        both = ev(h, n, blk)
        h2 = list(bad_h); h2[k] = good_h[k]
        n2 = list(bad_n); n2[k] = good_n[k]
        print(f"{k:5d} | {both - base:+.4f}       | {ev(h2, bad_n, blk) - base:+.4f}    | {ev(bad_h, n2, blk) - base:+.4f}", flush=True)


if __name__ == "__main__":
    main()
