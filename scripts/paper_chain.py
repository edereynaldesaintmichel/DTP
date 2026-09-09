"""Write the sequential run chain for the workshop-paper experiments of one
model (delta = 1, L = 4): seeds, score-vs-outcome layouts, ablations, and
optionally the two 10k-step runs. Every run is skipped if its log already has
the final step, so the chain can be relaunched after a crash.

  python scripts/paper_chain.py --model-id h2oai/h2o-danube3-500m-base --name danube3 --n-layers 16 > chain_danube3.sh
  (setsid nohup bash chain_danube3.sh > runs/paper/danube3/chain.out 2>&1 < /dev/null &)
  --gpus 0 1 splits the training runs across two GPUs (setup stays sequential; each lane logs to lane<i>.out).

Studies (memory: workshop-paper-experiment-plan):
  1 seeds       optimised x3 / contiguous x3 training seeds
  2 score->KL   random0..7, anti, k4/k8/k12 layers, heads-only, neurons-only, optimised, contiguous
  4 ablations   heads-only, neurons-only, add score (single seed)
  3 long        --long: optimised vs contiguous for --long-steps
"""

import argparse

TRAIN = ("python scripts/finetune.py --model-id {model} --devices 4 --delta 1 --distill --ce-weight 0.1 "
         "--micro-batch {mb} --grad-accum {ga} --lr 5e-5 --warmup 100 --min-lr-ratio 0.1 --freeze-embed "
         "--token-file {tokens} --token-skip 64 --eval-every {eval_every} --eval-blocks 16 --no-save "
         "--steps {steps} --seed {seed} --out {root}/{name} {perm}")


def spaced(k, n):
    return ",".join(str(round(j * n / k)) for j in range(k))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--name", required=True, help="short model name, used in paths")
    p.add_argument("--n-layers", type=int, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--random-layouts", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--long", action="store_true", help="append the two long runs")
    p.add_argument("--long-steps", type=int, default=10000)
    p.add_argument("--tokens", type=int, default=None, help="token file size; default sized for the longest run")
    p.add_argument("--venv", default="/venv/main/bin/activate")
    p.add_argument("--gpus", type=int, nargs="*", default=None,
                   help="GPU ids; with several, runs are dealt round-robin into one lane per GPU and the lanes run in parallel")
    a = p.parse_args()

    root = f"runs/paper/{a.name}"
    tokens = f"runs/tokens/{a.name}.npy"
    perms = f"{root}/perms"
    stats = f"{root}/affinity_stats.pt"
    seqs = a.micro_batch * a.grad_accum
    n_tok = a.tokens or int((max(a.steps, a.long_steps if a.long else 0) * seqs + 64) * 1025 * 1.05)

    out = ["#!/bin/bash", "set -e", 'cd "$(dirname "$0")"  # the script lives in the repo root',
           f"[ -f {a.venv} ] && source {a.venv}", "export OMP_NUM_THREADS=4",
           f"mkdir -p {root} runs/tokens", 'log() { echo "=== $1 $(date -u)"; }', ""]
    out += [f"[ -f {tokens} ] || {{ log pretokenize; python scripts/pretokenize.py --model-id {a.model_id} "
            f"--tokens {n_tok} --out {tokens}; }}",
            f"[ -f {stats} ] || {{ log affinity; python scripts/affinity_stats.py --model-id {a.model_id} "
            f"--token-file {tokens} --out {stats}; }}", ""]
    ex = f"python scripts/expertise.py --stats {stats} --model-id {a.model_id} --save-dir {perms} --perm-only --deltas 0 1 --n-blocks 16"
    layouts = [("optimised", f"--score fo --random-seeds {a.random_layouts}"), ("anti", "--score fo --minimise --random-seeds 0"),
               ("heads", "--score fo --heads-only --random-seeds 0"), ("neurons", "--score fo --neurons-only --random-seeds 0"),
               ("add", "--score add --random-seeds 0")]
    layouts += [(f"k{k}", f"--score fo --layers {spaced(k, a.n_layers)} --random-seeds 0") for k in (4, 8, 12)]
    for tag, flags in layouts:
        out.append(f"[ -f {perms}/{tag}.perm.pt ] || {{ log layout-{tag}; {ex} --tag {tag} {flags}; }}")
    out.append(f"python scripts/expertise.py score --stats {stats} --score fo {perms}/*.perm.pt > {root}/layout_scores.csv")
    out.append("")
    out.append("run() {  # name steps seed [perm]")
    out.append(f'  if [ -f {root}/$1/log.csv ] && grep -q "^$2," {root}/$1/log.csv; then echo "skip $1"; return; fi')
    out.append('  log "train $1"')
    out.append("  " + TRAIN.format(model=a.model_id, mb=a.micro_batch, ga=a.grad_accum, tokens=tokens,
                                   eval_every="$5", steps="$2", seed="$3", root=root, name="$1", perm="$4"))
    out.append("}")
    out.append("")
    ev = 100
    runs = []
    for s in a.seeds:  # study 1
        runs.append(f"run opt_s{s} {a.steps} {s} \"--init-perm {perms}/optimised.perm.pt\" {ev}")
        runs.append(f"run contig_s{s} {a.steps} {s} \"\" {ev}")
    for r in range(a.random_layouts):  # study 2
        runs.append(f"run random{r} {a.steps} {100 + r} \"--init-perm {perms}/random{r}.perm.pt\" {ev}")
    for tag in ["anti", "k4", "k8", "k12", "heads", "neurons", "add"]:  # study 2 spread + study 4
        runs.append(f"run {tag} {a.steps} {a.seeds[0]} \"--init-perm {perms}/{tag}.perm.pt\" {ev}")
    if a.long:  # study 3
        runs.append(f"run opt_long {a.long_steps} {a.seeds[0]} \"--init-perm {perms}/optimised.perm.pt\" 250")
        runs.append(f"run contig_long {a.long_steps} {a.seeds[0]} \"\" 250")
    if a.gpus and len(a.gpus) > 1:
        for i, g in enumerate(a.gpus):
            out += [f"lane{i}() {{", f"  export CUDA_VISIBLE_DEVICES={g}"] + ["  " + r for r in runs[i::len(a.gpus)]] + ["}", ""]
        out += [f"lane{i} > {root}/lane{i}.out 2>&1 &" for i in range(len(a.gpus))] + ["wait"]
    else:
        out += runs
    out += ["", "log done"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
