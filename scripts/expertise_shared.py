"""Layouts for the shared-expert DTP model (dtp/shared_model.py).

Per layer: pick the n_shared neurons to replicate on every device, then run the
usual expertised assignment (scripts/expertise.py) on the remaining neurons.
Shared neurons are marked with device id L in the saved *.perm.pt files.

Which neurons to share (--shared-by):
  dn : neurons whose output matters most to the next layer's attention
       (row mass of the down edge; the last layer falls back to fo). Ranks the
       layer-2 massive neurons last: their tokens have a huge rms, so in
       normalised units they look small.
  fo : neurons with the largest first-order damage (column mass of S_up_fo). Default.

    python scripts/expertise_shared.py --shared-frac 0.10 --score fo --save-dir runs/perms_shared
    python scripts/finetune.py ... --shared 308 --init-perm runs/perms_shared/optimised.perm.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.shared_model import check_layout, mark_shared, n_shared_for
from scripts.expertise import Chain, random_partition


def choose_shared(o, i, n_shared, by):
    """Indices of the n_shared neurons of layer i to replicate."""
    if by == "dn" and i + 1 < o["NL"]:
        mass = o["S_dn_add"][i].sum(1)  # [I] how much next-layer q/k/v reads from each neuron
    else:
        mass = o["S_up_fo"][i].sum(0)  # [I] first-order residual damage per neuron
    return mass.topk(n_shared).indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", default="runs/affinity_stats.pt")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--shared-frac", type=float, default=0.10)
    p.add_argument("--shared-by", default="fo", choices=["dn", "fo"])
    p.add_argument("--score", default="fo", choices=["add", "abl", "fo"], help="up-edge score for the sharded part")
    p.add_argument("--dn-score", default="add", choices=["add", "abl"])
    p.add_argument("--random-seeds", type=int, default=2)
    p.add_argument("--save-dir", required=True)
    args = p.parse_args()

    L = args.devices
    o = torch.load(args.stats)
    NL, KV, hpg, I = o["NL"], o["KV"], o["hpg"], o["I"]
    n_shared = n_shared_for(I, L, args.shared_frac)
    print(f"sharing {n_shared} of {I} neurons per layer ({n_shared / I:.1%}); {(I - n_shared) // L} sharded per device")
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    shared = [choose_shared(o, i, n_shared, args.shared_by) for i in range(NL)]
    keep = [torch.ones(I, dtype=torch.bool).index_fill_(0, s, False).nonzero().flatten() for s in shared]
    for i in (2,):
        print(f"layer {i} shared neurons (top 10): {shared[i][:10].tolist()}")

    # edge matrices restricted to the sharded neurons, normalised to unit mass per layer
    S_up = [o[f"S_up_{args.score}"][i].view(KV, hpg, -1).sum(1)[:, keep[i]].double().to(dev_) for i in range(NL)]
    S_dn = [o[f"S_dn_{args.dn_score}"][i][keep[i]].double().to(dev_) for i in range(NL - 1)]
    S_up = [s / s.sum() for s in S_up]
    S_dn = [s / s.sum().clamp_min(1e-30) for s in S_dn]
    chain = Chain(S_up, S_dn, L)

    ident_h = [(torch.arange(KV) // (KV // L)).to(dev_) for _ in range(NL)]
    print("optimising the sharded part...")
    head_dev, neur_kept, obj = chain.solve(ident_h)
    n_edges = 2 * NL - 1
    print(f"  co-located fraction per edge: {obj / n_edges:.3f} (random {1 / L:.3f})")

    def full_layout(neur_kept_i, keep_i, shared_i):
        nd = torch.full((I,), L, dtype=torch.long)
        nd[keep_i] = neur_kept_i.cpu()
        nd = mark_shared(nd, shared_i, L)
        check_layout(nd, L, n_shared)
        return nd

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    neur_dev = [full_layout(neur_kept[i], keep[i], shared[i]) for i in range(NL)]
    torch.save(dict(head_dev=[h.cpu() for h in head_dev], neur_dev=neur_dev, n_shared=n_shared), out / "optimised.perm.pt")
    print(f"saved {out / 'optimised.perm.pt'}")

    # random baselines with the same number of (randomly chosen) shared neurons
    for s in range(args.random_seeds):
        gen = torch.Generator().manual_seed(1000 + s)
        hs, ns = [], []
        for i in range(NL):
            h, n = random_partition(KV, I - n_shared, L, gen)
            perm = torch.randperm(I, generator=gen)
            nd = torch.full((I,), L, dtype=torch.long)
            nd[perm[n_shared:]] = n
            check_layout(nd, L, n_shared)
            hs.append(h)
            ns.append(nd)
        torch.save(dict(head_dev=hs, neur_dev=ns, n_shared=n_shared), out / f"random{s}.perm.pt")
    print(f"saved {args.random_seeds} random layouts")


if __name__ == "__main__":
    main()
