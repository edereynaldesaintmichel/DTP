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
- `tests/test_dtp.py`
