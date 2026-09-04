"""Expertised sharding for DTP at delta = 1: choose, per layer, which KV groups
and which FFN neurons go to which virtual device so that the co-located
affinity (from affinity_stats.py) is maximal, apply it as a pure permutation of
the HF weights (the vanilla function is unchanged), and evaluate DTP ppl at a
few delays against random balanced permutations.

Objective (maximised): sum over layers of
    sum_{KV group g, neuron n on the same device} S_up[g, n]
  + sum_{neuron n, next-layer KV group g on the same device} S_dn[n, g]
Each edge matrix is normalised to unit mass per layer (--no-normalize to use raw
units), so the objective reads as "number of edges times co-located fraction".

Optimisation: coordinate ascent on the chain. Neurons given both neighbouring
head partitions: exact balanced assignment (linear_sum_assignment on expanded
slots). Heads given both neighbouring neuron assignments: exhaustive over all
labelled balanced partitions of the KV groups (2520 for 8 groups / 4 devices).
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import perplexity, wikitext_blocks
from dtp.model import DTPQwen3


def labelled_partitions(G, L):
    """All assignments of G groups to L devices with G/L each. [P, G] int."""
    base = [l for l in range(L) for _ in range(G // L)]
    return torch.tensor(sorted(set(itertools.permutations(base))), dtype=torch.long)


def assign_neurons(score, L, iters=300):
    """score: [I, L] gain of putting neuron i on device l. Balanced assignment
    (I/L per device) by dual ascent on L offsets (GPU), a margin fix-up to hit
    the capacities exactly, then greedy pairwise swaps until no swap improves.
    Returns [I] device ids (on score's device)."""
    I = score.shape[0]
    cap = I // L
    lam = torch.zeros(L, dtype=score.dtype, device=score.device)
    step = score.std().item() + 1e-12
    for _ in range(iters):
        dev = (score + lam).argmax(1)
        cnt = torch.bincount(dev, minlength=L)
        if (cnt == cap).all():
            break
        lam = lam - step * (cnt - cap).to(score.dtype) / cap
        step *= 0.98
    # fix-up: move lowest-loss neurons out of overfull devices
    while True:
        cnt = torch.bincount(dev, minlength=L)
        over = (cnt > cap).nonzero().flatten()
        if len(over) == 0:
            break
        under = (cnt < cap).nonzero().flatten()
        o = over[0]
        adj = score + lam
        alt = adj[:, under].max(1)
        loss = adj[torch.arange(I, device=score.device), dev] - alt.values
        loss[dev != o] = float("inf")
        i = loss.argmin()
        dev[i] = under[alt.indices[i]]
    # 2-opt swaps
    while True:
        best_gain, best = 0.0, None
        for a in range(L):
            for b in range(a + 1, L):
                ia = (dev == a).nonzero().flatten()
                ib = (dev == b).nonzero().flatten()
                ga = (score[ia, b] - score[ia, a])
                gb = (score[ib, a] - score[ib, b])
                va, ka = ga.max(0)
                vb, kb = gb.max(0)
                if va + vb > best_gain + 1e-12:
                    best_gain, best = (va + vb).item(), (ia[ka], ib[kb], a, b)
        if best is None:
            break
        i, j, a, b = best
        dev[i], dev[j] = b, a
    return dev


def onehot(dev, L):
    return torch.nn.functional.one_hot(dev, L).to(torch.float64)


class Chain:
    def __init__(self, S_up, S_dn, L):
        # S_up: list of [G, I] per layer; S_dn: list of [I, G_next] per layer (len NL-1)
        self.S_up, self.S_dn, self.L = S_up, S_dn, L
        self.NL = len(S_up)
        self.G = S_up[0].shape[0]
        self.I = S_up[0].shape[1]
        self.dev = S_up[0].device
        self.P = labelled_partitions(self.G, L).to(self.dev)  # [P, G]
        self.P1h = onehot(self.P, L)  # [P, G, L]

    def objective(self, head_dev, neur_dev):
        tot = 0.0
        for i in range(self.NL):
            Hh, Nn = onehot(head_dev[i], self.L), onehot(neur_dev[i], self.L)
            tot += (Hh.T @ self.S_up[i] @ Nn).diagonal().sum().item()
            if i + 1 < self.NL:
                Hn = onehot(head_dev[i + 1], self.L)
                tot += (Nn.T @ self.S_dn[i] @ Hn).diagonal().sum().item()
        return tot

    def neuron_score(self, i, head_dev):
        """[I, L] gain per neuron per device given head partitions."""
        s = self.S_up[i].T @ onehot(head_dev[i], self.L)  # [I, L]
        if i + 1 < self.NL:
            s = s + self.S_dn[i] @ onehot(head_dev[i + 1], self.L)
        return s

    def best_heads(self, i, neur_dev):
        """Exhaustive best labelled partition of layer i's KV groups given the
        neuron assignments of layers i (up edge) and i-1 (down edge)."""
        M = self.S_up[i] @ onehot(neur_dev[i], self.L)  # [G, L] gain of putting group g on device l
        if i > 0:
            M = M + self.S_dn[i - 1].T @ onehot(neur_dev[i - 1], self.L)
        scores = (self.P1h * M[None]).sum((1, 2))  # [P]
        return self.P[scores.argmax()].clone()

    def solve(self, head_init, verbose=True, max_sweeps=20):
        head_dev = [h.clone() for h in head_init]
        neur_dev = [assign_neurons(self.neuron_score(i, head_dev), self.L) for i in range(self.NL)]
        best = self.objective(head_dev, neur_dev)
        if verbose:
            print(f"  init (given heads): {best:.4f}")
        for sweep in range(max_sweeps):
            for i in range(self.NL):
                head_dev[i] = self.best_heads(i, neur_dev)
                neur_dev[i] = assign_neurons(self.neuron_score(i, head_dev), self.L)
            cur = self.objective(head_dev, neur_dev)
            if verbose:
                print(f"  sweep {sweep}: {cur:.4f}")
            if cur <= best + 1e-9:
                break
            best = cur
        return head_dev, neur_dev, best


# ------------------------------------------------------------- permutation

@torch.no_grad()
def apply_permutation(hf, head_dev, neur_dev, L):
    """Reorder heads/neurons so that device l's contiguous shard (as sliced by
    DTPQwen3's views) holds the groups/neurons assigned to l. Pure permutation."""
    cfg = hf.config
    H, KV, D, I = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size, cfg.intermediate_size
    hpg = H // KV
    for i, layer in enumerate(hf.model.layers):
        at, mlp = layer.self_attn, layer.mlp
        hd = at.head_dim
        g_order = torch.argsort(head_dev[i].cpu(), stable=True)  # new position -> old group
        q_idx = (g_order[:, None] * (hpg * hd) + torch.arange(hpg * hd)[None]).flatten()
        kv_idx = (g_order[:, None] * hd + torch.arange(hd)[None]).flatten()
        at.q_proj.weight.copy_(at.q_proj.weight[q_idx].clone())
        at.k_proj.weight.copy_(at.k_proj.weight[kv_idx].clone())
        at.v_proj.weight.copy_(at.v_proj.weight[kv_idx].clone())
        at.o_proj.weight.copy_(at.o_proj.weight[:, q_idx].clone())
        n_order = torch.argsort(neur_dev[i].cpu(), stable=True)
        mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[n_order].clone())
        mlp.up_proj.weight.copy_(mlp.up_proj.weight[n_order].clone())
        mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, n_order].clone())


def random_partition(G, I, L, gen, device="cpu"):
    head = torch.randperm(G, generator=gen) % L  # balanced: G/L per device
    neur = torch.randperm(I, generator=gen) % L
    return head.to(device), neur.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", default="runs/affinity_stats.pt")
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--score", default="add", choices=["add", "abl", "fo"], help="up-edge score")
    p.add_argument("--dn-score", default="add", choices=["add", "abl"], help="down-edge score")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--up-only", action="store_true", help="ignore the FFN -> next attention edge")
    p.add_argument("--deltas", type=float, nargs="*", default=[0, 1, 2, 4])
    p.add_argument("--random-seeds", type=int, default=5)
    p.add_argument("--n-blocks", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    p.add_argument("--save-dir", default=None, help="save permuted HF state dicts (optimised + randoms) here")
    p.add_argument("--check-lsa", action="store_true", help="compare the GPU assignment with scipy on layer 5")
    args = p.parse_args()

    L = args.devices
    o = torch.load(args.stats)
    NL, KV, hpg = o["NL"], o["KV"], o["hpg"]
    dev_ = "cuda"
    S_up = [o[f"S_up_{args.score}"][i].view(KV, hpg, -1).sum(1).double().to(dev_) for i in range(NL)]
    S_dn = [o[f"S_dn_{args.dn_score}"][i].double().to(dev_) for i in range(NL - 1)]
    if args.up_only:
        S_dn = [torch.zeros_like(s) for s in S_dn]
    if not args.no_normalize:
        S_up = [s / s.sum() for s in S_up]
        S_dn = [s / s.sum().clamp_min(1e-30) for s in S_dn]
    chain = Chain(S_up, S_dn, L)
    n_edges = NL + (0 if args.up_only else NL - 1)

    # ---- objective: identity, random, optimised
    ident_h = [(torch.arange(KV) // (KV // L)).to(dev_) for _ in range(NL)]
    ident_n = [(torch.arange(o["I"]) // (o["I"] // L)).to(dev_) for _ in range(NL)]
    if args.check_lsa:
        sc = chain.neuron_score(5, ident_h)
        d_gpu = assign_neurons(sc, L)
        cost = -sc.repeat_interleave(o["I"] // L, dim=1).cpu().numpy()
        r, c = linear_sum_assignment(cost)
        d_lsa = torch.zeros(o["I"], dtype=torch.long); d_lsa[r] = torch.from_numpy(c) // (o["I"] // L)
        v_gpu = sc[torch.arange(o["I"], device=dev_), d_gpu].sum().item()
        v_lsa = sc.cpu()[torch.arange(o["I"]), d_lsa].sum().item()
        print(f"assignment check layer 5: gpu {v_gpu:.6f}  scipy {v_lsa:.6f}")
    obj_id = chain.objective(ident_h, ident_n)
    rand_parts, obj_rand = [], []
    for s in range(args.random_seeds):
        gen = torch.Generator().manual_seed(1000 + s)
        parts = [random_partition(KV, o["I"], L, gen, dev_) for _ in range(NL)]
        rand_parts.append(([h for h, _ in parts], [n for _, n in parts]))
        obj_rand.append(chain.objective(*rand_parts[-1]))
    print(f"co-located affinity (sum over {n_edges} edges; random expectation {n_edges / L:.2f}):")
    print(f"  identity {obj_id:.3f}  random {np.mean(obj_rand):.3f} +- {np.std(obj_rand):.3f}")
    print("optimising...")
    head_dev, neur_dev, obj_opt = chain.solve(ident_h)
    print(f"  optimised {obj_opt:.3f}  ({obj_opt / n_edges:.3f} co-located fraction per edge, random {1 / L:.3f})")
    per_layer = []
    for i in range(NL):
        up = (onehot(head_dev[i], L).T @ S_up[i] @ onehot(neur_dev[i], L)).diagonal().sum().item() / S_up[i].sum().item()
        per_layer.append(up)
    print("  per-layer co-located fraction of the up edge: " + " ".join(f"{v:.2f}" for v in per_layer))

    # ---- evaluate
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    blocks = wikitext_blocks(tok, args.seq_len, args.n_blocks)
    base_sd = Qwen3ForCausalLM.from_pretrained(args.model_id, dtype=dtype).state_dict()
    hf = Qwen3ForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device).eval()
    dtp = DTPQwen3(hf, n_devices=L, delta=0).to(device).eval()

    def eval_all(tag):
        row = {}
        for d in args.deltas:
            dtp.delta = d
            ppl, nll = perplexity(lambda x: dtp(x).logits, blocks, device, 4)
            row[d] = (ppl, nll)
        print(f"  {tag:<14s} " + "  ".join(f"d={d:g}: ppl {row[d][0]:8.3f} nll {row[d][1]:.4f}" for d in args.deltas), flush=True)
        return row

    def reset():
        hf.load_state_dict(base_sd)

    # sanity: permutation leaves the vanilla model unchanged
    x0 = blocks[:1, :-1].to(device)
    ref = hf(x0).logits.float()
    apply_permutation(hf, head_dev, neur_dev, L)
    diff = (hf(x0).logits.float() - ref).abs().max().item()
    print(f"vanilla logits max |diff| after permutation: {diff:.2e}")
    reset()

    print("wikitext-2 ppl:")
    results = {}
    results["identity"] = eval_all("identity")
    sd_dir = Path(args.save_dir) if args.save_dir else None
    if sd_dir:
        sd_dir.mkdir(parents=True, exist_ok=True)
    for s, (h, n) in enumerate(rand_parts):
        apply_permutation(hf, h, n, L)
        results[f"random{s}"] = eval_all(f"random{s}")
        if sd_dir:
            torch.save(hf.state_dict(), sd_dir / f"random{s}.pt")
            torch.save(dict(head_dev=[t.cpu() for t in h], neur_dev=[t.cpu() for t in n]), sd_dir / f"random{s}.perm.pt")
        reset()
    apply_permutation(hf, head_dev, neur_dev, L)
    results["optimised"] = eval_all("optimised")
    if sd_dir:
        torch.save(hf.state_dict(), sd_dir / "optimised.pt")
        torch.save(dict(head_dev=[t.cpu() for t in head_dev], neur_dev=[t.cpu() for t in neur_dev]), sd_dir / "optimised.perm.pt")

    print("\nsummary (nll):")
    for d in args.deltas:
        r = [results[f"random{s}"][d][1] for s in range(args.random_seeds)]
        print(f"  delta={d:g}: identity {results['identity'][d][1]:.4f}  random {np.mean(r):.4f} +- {np.std(r):.4f}  optimised {results['optimised'][d][1]:.4f}")


if __name__ == "__main__":
    main()
