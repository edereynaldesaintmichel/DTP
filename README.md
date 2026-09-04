# Delayed Tensor Parallelism for Qwen3-0.6B

Implementation of [Kog's Delayed Tensor Parallelism (DTP)](https://blog.kog.ai/delayed-tensor-parallelism-for-faster-transformer-inference/)
applied to `Qwen/Qwen3-0.6B-Base`, designed to run on a single RTX 5090.

## What DTP is

Standard tensor parallelism shards every attention/MLP module across L devices
and **all-reduces the partial outputs after every module** — 2·N_L blocking
syncs per forward pass. DTP removes the blocking sync: each device adds its
*own* partial output to its residual stream immediately, broadcasts it, and
folds in the *other* devices' partials **δ modules later**, when the transfer
has had δ modules' worth of weight-streaming time to complete. With module
index n (2 modules per layer: attention, then MLP):

- **Stage 1** (n < δ): `X_l ← X_l + √L · o_l^n` (√L mimics the magnitude of a full all-reduce)
- **Stage 2** (δ ≤ n < 2N_L − δ): `X_l ← X_l + o_l^n + Σ_{j≠l} o_j^{n−δ}`
- **Stage 3** (last δ modules): no new broadcasts; in-flight aggregations still land

Device residual streams intentionally diverge. At the end, the final RMSNorm and
LM head are applied per device and the L outputs are averaged (the final
all-reduce). Because the LM head is linear, we average the normed hidden states
and run the head once — mathematically identical.

This changes the function the network computes, so it is an **architecture
change that requires (up)training** — the blog trains from scratch and recovers
near-vanilla loss. Here we start from pretrained Qwen3-0.6B-Base weights,
measure the zero-shot damage as a function of δ, then fine-tune to claw quality
back.

## Why Qwen3

Qwen3 is a textbook **pre-norm** transformer: each module's output is added to
the residual raw, so the per-device partials sum exactly to the vanilla module
output and the blog's equations apply verbatim. (Gemma 3, the previous target,
has sandwich post-norms that act on the full module output — which no device
holds under DTP — forcing an ad-hoc per-partial norm approximation.) Its 16
query heads / 8 KV heads / 3072 MLP width shard evenly for **L ∈ {2, 4, 8}**,
and the model is ungated on Hugging Face.

## What this repo does and does not show

A single GPU cannot demonstrate the *speedup* (there is no real inter-device
communication to hide). This repo simulates L **virtual devices** on one GPU to
study the *quality* side: how much the delayed aggregation hurts, and how much
fine-tuning recovers — the same experiment as the blog's Figures 5–6, but
starting from pretrained weights. The sharding math is real (partial outputs
sum exactly to the vanilla module output, verified by tests), so the code is a
correct reference for a later multi-GPU port: replace the in-tensor device axis
with NCCL ranks and the queue pops with async all-reduce waits.

## Design decisions the blog leaves open

- **Stage 3 own-output scale**: the blog is ambiguous; default is `√L`
  (magnitude-consistent, since the missing cross-device terms never arrive),
  switchable with `--stage3-scale one`.
- δ is counted **in modules** (2 per layer; Qwen3-0.6B has 28 layers →
  56 modules, so δ ≤ 28).

All shard weights are *views* into the wrapped HF model's parameters — no
duplication, and fine-tuned checkpoints remain valid `Qwen3ForCausalLM` state
dicts.

## Setup (vast.ai RTX 5090)

```bash
git clone <this repo> && cd DTP        # or rsync the directory
bash setup_vast.sh                     # torch cu128, transformers v5, runs CPU tests
```

Qwen3-0.6B-Base is ungated — no Hugging Face login needed.

## Run

**1. Zero-shot damage sweep** (wikitext-2 perplexity vs δ, ~2 min):

```bash
python scripts/eval_ppl.py --devices 4 --deltas 0 1 2 4 8 --vanilla
```

Expect `vanilla` ≈ low tens; DTP without fine-tuning will be much worse and
grow with δ — that gap is what uptraining removes.

**2. Fine-tune DTP** (defaults: L=4, δ=4, 2000 steps × 64 seqs × 1024 tokens
≈ 131M tokens, bf16 autocast + gradient checkpointing; roughly 2–3 h on a 5090):

```bash
python scripts/finetune.py --devices 4 --delta 4 --steps 2000 --out runs/dtp_l4_d4
```

**3. Fair baseline** — fine-tune the *vanilla* model on the same data/budget, so
"DTP damage" is separated from ordinary fine-tuning drift:

```bash
python scripts/finetune.py --vanilla --steps 2000 --out runs/vanilla_baseline
```

**4. Re-evaluate and compare:**

```bash
python scripts/eval_ppl.py --devices 4 --deltas 4 --checkpoint runs/dtp_l4_d4/model_state.pt
python scripts/eval_ppl.py --vanilla --deltas --checkpoint runs/vanilla_baseline/model_state.pt
```

**5. Sample text** from the adapted model:

```bash
python scripts/generate.py --devices 4 --delta 4 \
    --checkpoint runs/dtp_l4_d4/model_state.pt \
    --prompt "The Eiffel Tower" --temperature 0.8
```

Interesting sweeps: `--delta {1,2,4,8}` × `--devices {2,4,8}` (fine-tune each),
and `--stage3-scale one` — mirroring the blog's exposed-wait-time-vs-loss
trade-off, with δ as the knob.

## Expertised sharding (delta = 1)

The DTP layout is a free choice: which KV heads and which FFN neurons go to
which device. At delta = 1 a module only misses the *other* devices' output of
the immediately preceding module, so the damage is smallest when each device's
FFN shard is the part of the FFN that reads most from that device's own heads,
and each device's next-layer heads read most from its own FFN shard. This is a
pure permutation of the HF weights (rows of q/k/v, gate, up; columns of o_proj
and down), so the dense model is unchanged. It is an initialisation only.

Pipeline, in order (all on one 5090, a few minutes end to end):

```bash
# 1. affinity scores: how much each FFN neuron's output (through the down
#    projection) depends on each KV group, and how much the next layer's
#    q/k/v read from each neuron. One forward pass, 64k tokens, no gradient.
python scripts/affinity_stats.py --out runs/affinity_stats.pt

# 2. layout: alternate exact assignment of neurons (balanced linear assignment)
#    and heads (exhaustive over the 2520 splits of 8 KV groups into 4 pairs),
#    apply the permutation, and report untrained DTP ppl vs random layouts.
python scripts/expertise.py --score fo --save-dir runs/perms

# 3. distil the DTP model from the dense one, starting from that layout
python scripts/finetune.py --devices 4 --delta 1 --distill --freeze-embed \
    --micro-batch 4 --grad-accum 4 --steps 2000 --warmup 100 --eval-every 100 \
    --eval-blocks 16 --no-save --init-perm runs/perms/optimised.perm.pt --out runs/train_fo
# same command with --init-perm runs/perms/random1.perm.pt for the baseline
```

Results (Qwen3-0.6B-Base, L = 4, delta = 1, same recipe for every layout;
logs and the figure are in `runs/expertise_logs/`):

| step | expertised init: ppl / KL | random sharding: ppl / KL |
|---|---|---|
| 0 (no training) | 608 / 3.85 | 3088 / 5.51 |
| 500 | 18.87 / 0.398 | 20.95 / 0.501 |
| 1000 | 17.37 / 0.322 | 18.82 / 0.404 |
| 2000 | 16.42 / 0.265 | 17.82 / 0.346 |

The expertised init reaches the random layout's final perplexity at step 814,
i.e. 2.5x fewer steps. At 500 steps the dot-product score (`--score add`)
gives 19.31 and the first-order score (`--score fo`) 18.33, vs 20.8 and 21.6
for two random seeds.

Code, in reading order:

- `scripts/affinity_stats.py` — the three affinity scores (`add`, `abl`, `fo`)
  for the two edges per layer (heads -> same-layer neurons, neurons -> next-layer heads)
- `scripts/expertise.py` — `Chain` (the alternating optimiser), `apply_permutation`,
  random baselines, untrained evaluation
- `scripts/finetune.py` — `--init-perm` applies a saved layout before training
- `dtp/shared_model.py`, `scripts/expertise_shared.py` — variant with a "shared
  expert": 10% of each layer's neurons replicated on every device, added locally
  and never broadcast (`--shared 308` in finetune.py). 16.26 / 0.260 at 2000
  steps, i.e. within noise of the plain expertised init.
- diagnostics behind the "where the gain comes from" story: `scripts/massive_probe.py`
  (the layer-2 massive-activation neurons and the heads that drive them),
  `scripts/colocate_test.py` (co-locating them with their KV group, everything
  else fixed), `scripts/random_diag.py` (why random layouts differ so much untrained)

## Tests

```bash
python -m pytest tests -q
```

CPU, seconds, no model download (tiny random Qwen3 config). They pin down:
exact logit equivalence with Hugging Face at L=1, δ=0; per-device partials
summing exactly to the vanilla module outputs; the residual/queue bookkeeping
against a literal per-device transcription of the blog's three-stage equations
for δ ∈ {0,1,3,4}; KV-cache decoding matching the full-sequence forward;
gradient-checkpointed loss/grads matching the plain path.

## Files

- `dtp/model.py` — `DTPQwen3` wrapper: sharding, delay queue, cache, generate
- `dtp/data.py` — streamed packed fine-tuning data, wikitext-2 perplexity
- `scripts/eval_ppl.py`, `scripts/finetune.py`, `scripts/generate.py`
- `scripts/affinity_stats.py`, `scripts/expertise.py`, `scripts/expertise_shared.py`,
  `dtp/shared_model.py` — expertised sharding, see the section above
- `tests/test_dtp.py`
