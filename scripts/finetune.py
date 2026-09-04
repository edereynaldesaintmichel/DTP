"""Uptrain Qwen3-0.6B-Base under Delayed Tensor Parallelism (or the vanilla
model, for a fair drift baseline) on streamed FineWeb-Edu.

Typical runs:
  # recover DTP by distilling from the frozen original model (recommended)
  python scripts/finetune.py --devices 4 --delta 4 --distill --freeze-embed \
      --micro-batch 4 --grad-accum 16 --steps 2000 --out runs/dtp_l4_d4_kd
  # annealed continuous delta: ramps 0 -> ~10 nominal over 2000 steps
  # (quadratic for 100 steps, then 0.005/step; crosses 4 around step 850),
  # pausing whenever eval ppl exceeds --gate-ppl so training catches up
  python scripts/finetune.py --devices 4 --distill --freeze-embed --delta-schedule \
      --micro-batch 4 --grad-accum 16 --steps 2000 --out runs/dtp_l4_anneal_kd
  # one-hot CE variants (need the --vanilla drift baseline for a fair comparison)
  python scripts/finetune.py --devices 4 --delta 4 --steps 2000 --out runs/dtp_l4_d4
  python scripts/finetune.py --vanilla --steps 2000 --out runs/vanilla_baseline
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import batched, packed_stream, perplexity, wikitext_blocks
from dtp.model import DTPQwen3


def kd_loss(student_logits, teacher_logits, labels, ce_weight, chunk=256):
    """Per-token KL(teacher || student) + ce_weight * CE. Chunked over the
    sequence with activation checkpointing: at Qwen3's 152k vocab a full-length
    fp32 log-softmax pair is ~5 GB per micro-batch of 4, so it is recomputed
    chunk-by-chunk in backward instead of kept alive."""
    from torch.utils.checkpoint import checkpoint

    def piece(sl, tl, lb):
        logq = F.log_softmax(sl.float(), -1)
        logp = F.log_softmax(tl.float(), -1)
        kl = (logp.exp() * (logp - logq)).sum(-1).sum()
        ce = F.nll_loss(logq.reshape(-1, logq.shape[-1]), lb.reshape(-1), reduction="sum")
        return kl + ce_weight * ce

    total = 0.0
    for i in range(0, labels.shape[1], chunk):
        total = total + checkpoint(
            piece, student_logits[:, i : i + chunk], teacher_logits[:, i : i + chunk],
            labels[:, i : i + chunk], use_reentrant=False,
        )
    return total / labels.numel()


class DeltaScheduler:
    """Continuous-delta ramp: quadratic for the first `ramp` scheduler steps
    (y = slope/(2*ramp) * s^2), then linear (y = slope * (s - ramp/2)) — C1 at
    the joint — capped at `cap`. The clock `s` advances only while the last
    gate check passed (eval ppl <= gate_ppl), so the ramp pauses until training
    catches up and resumes where it left off."""

    def __init__(self, slope, ramp, cap, gate_ppl):
        self.slope, self.ramp, self.cap, self.gate_ppl = slope, ramp, cap, gate_ppl
        self.s = 0
        self.open = True

    def gate(self, ppl):
        self.open = ppl <= self.gate_ppl
        return self.open

    @property
    def value(self):
        s = self.s
        y = self.slope * s * s / (2 * self.ramp) if s <= self.ramp else self.slope * (s - self.ramp / 2)
        return min(y, self.cap)

    def step(self):
        if self.open:
            self.s += 1
        return self.value


def lr_at(step, base, warmup, total, min_ratio):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--delta", type=float, default=4)
    p.add_argument("--delta-schedule", action="store_true",
                   help="anneal delta from 0 (quadratic ramp then linear), gated on eval ppl")
    p.add_argument("--delta-slope", type=float, default=0.005,
                   help="delta growth per ungated step in the linear regime")
    p.add_argument("--delta-ramp", type=int, default=100,
                   help="length of the initial quadratic ramp, in scheduler steps")
    p.add_argument("--delta-max", type=float, default=None,
                   help="cap for the annealed delta (default: the model limit, n_layers)")
    p.add_argument("--gate-ppl", type=float, default=20.0,
                   help="pause the delta ramp while quick eval ppl exceeds this")
    p.add_argument("--gate-every", type=int, default=25,
                   help="steps between quick gate-ppl checks")
    p.add_argument("--gate-blocks", type=int, default=8,
                   help="eval blocks used for the quick gate-ppl check")
    p.add_argument("--stage3-scale", default="sqrt_l", choices=["sqrt_l", "one"])
    p.add_argument("--vanilla", action="store_true", help="fine-tune the stock model instead (baseline)")
    p.add_argument("--distill", action="store_true",
                   help="distill from a frozen copy of the original model instead of one-hot CE")
    p.add_argument("--ce-weight", type=float, default=0.1,
                   help="weight of the one-hot CE term added to the KD loss")
    p.add_argument("--kd-chunk", type=int, default=256, help="sequence chunk for the KD loss")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (and teacher)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--freeze-embed", action="store_true",
                   help="freeze the (tied) 156M-param embedding/LM head")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-blocks", type=int, default=32)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--out", default="runs/dtp")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--init-state", default=None,
                   help="state dict to load into the student before training (e.g. a permuted model)")
    p.add_argument("--init-perm", default=None,
                   help="head/neuron device assignment (from expertise.py --save-dir, *.perm.pt) applied "
                        "to the student as a pure permutation before training")
    p.add_argument("--no-save", action="store_true", help="do not write model_state.pt")
    p.add_argument("--shared", type=int, default=0,
                   help="shared-expert size: FFN neurons per layer replicated on every device (dtp/shared_model.py); "
                        "the layout from --init-perm must mark them with device id = --devices")
    args = p.parse_args()

    assert torch.cuda.is_available(), "finetune.py expects a CUDA GPU"
    device = "cuda"
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model_id)
    # fp32 master weights + bf16 autocast
    hf = Qwen3ForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
    if args.init_state:
        hf.load_state_dict(torch.load(args.init_state, map_location="cpu", weights_only=True))
    if args.init_perm:
        from scripts.expertise import apply_permutation

        perm = torch.load(args.init_perm, weights_only=True)
        apply_permutation(hf, perm["head_dev"], perm["neur_dev"], args.devices)
    hf = hf.to(device)

    if args.vanilla:
        assert not args.distill, "--distill is for repairing the DTP model"
        hf.gradient_checkpointing_enable()
        hf.train()
        model = hf

        def loss_fn(x):
            return model(input_ids=x, labels=x).loss

        def eval_logits(x):
            return hf(x).logits
    else:
        if args.shared:
            from dtp.shared_model import SharedDTPQwen3

            model = SharedDTPQwen3(
                hf, n_devices=args.devices, delta=args.delta, n_shared=args.shared,
                stage3_own_scale=args.stage3_scale, gradient_checkpointing=True,
            )
        else:
            model = DTPQwen3(
                hf, n_devices=args.devices, delta=args.delta,
                stage3_own_scale=args.stage3_scale, gradient_checkpointing=True,
            )
        model.train()

        def loss_fn(x):
            return model(x, labels=x).loss

        def eval_logits(x):
            return model(x).logits

    teacher = None
    if args.distill:
        # The DTP wrapper's weights are views into `hf`, which is being trained —
        # the teacher must be a separate frozen copy of the original weights.
        teacher = Qwen3ForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).to(device)
        teacher.eval().requires_grad_(False)

        def loss_fn(x):  # noqa: F811 — replaces the one-hot CE loss
            inp, lab = x[:, :-1], x[:, 1:]
            with torch.no_grad():
                t_logits = teacher(inp).logits
            return kd_loss(model(inp).logits, t_logits, lab, args.ce_weight, args.kd_chunk)

    sched = None
    if args.delta_schedule:
        assert not args.vanilla, "--delta-schedule requires the DTP model"
        assert not args.compile, "--delta-schedule changes delta every step, which would retrigger torch.compile"
        cap = args.delta_max if args.delta_max is not None else model.n_modules // 2
        sched = DeltaScheduler(args.delta_slope, args.delta_ramp, cap, args.gate_ppl)
        model.delta = sched.value  # start at 0: exactly the vanilla model

    if args.freeze_embed:
        hf.model.embed_tokens.weight.requires_grad_(False)

    if args.compile:
        # `hf` keeps pointing at the eager module, so state_dict saves stay clean.
        model = torch.compile(model)
        if teacher is not None:
            teacher = torch.compile(teacher)

    params = [p_ for p_ in hf.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay, fused=True)

    eval_blk = wikitext_blocks(tok, args.seq_len, args.eval_blocks)
    stream = batched(
        packed_stream(tok, args.seq_len, args.dataset, args.dataset_config,
                      shuffle_seed=args.seed),
        args.micro_batch,
    )
    log = open(out / "log.csv", "w")
    log.write("step,loss,lr,tok_per_s,eval_ppl,eval_kl,delta,gate_ppl\n")

    @torch.no_grad()
    def eval_kl():
        tot, n = 0.0, 0
        for i in range(0, len(eval_blk), 2):
            x = eval_blk[i : i + 2, :-1].to(device)
            logp = F.log_softmax(teacher(x).logits.float(), -1)
            logq = F.log_softmax(model(x).logits.float(), -1)
            tot += (logp.exp() * (logp - logq)).sum(-1).sum().item()
            n += logp.shape[0] * logp.shape[1]
        return tot / n

    def run_eval():
        model.eval()
        with torch.autocast("cuda", torch.bfloat16):
            ppl, _ = perplexity(eval_logits, eval_blk, device, micro_batch=4)
            kl = eval_kl() if teacher is not None else None
        model.train()
        return ppl, kl

    @torch.no_grad()
    def quick_gate_ppl():
        model.eval()
        with torch.autocast("cuda", torch.bfloat16):
            ppl, _ = perplexity(eval_logits, eval_blk[: args.gate_blocks], device, micro_batch=4)
        model.train()
        return ppl

    def fmt(ppl, kl):
        return f"eval ppl {ppl:.3f}" + (f"  KL {kl:.4f}" if kl is not None else "")

    ppl0, kl0 = run_eval()
    print(f"step 0: {fmt(ppl0, kl0)}")
    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    t0 = time.time()
    for step in range(args.steps):
        gate_p = None
        if sched is not None:
            if step and step % args.gate_every == 0:
                gate_p = quick_gate_ppl()
                sched.gate(gate_p)
            model.delta = sched.step()
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            batch = next(stream).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                loss = loss_fn(batch)
            (loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        tps = tokens_per_step * (step + 1) / (time.time() - t0)
        ppl_s, kl_s = "", ""
        if args.eval_every and (step + 1) % args.eval_every == 0:
            ppl, kl = run_eval()
            ppl_s, kl_s = f"{ppl:.3f}", "" if kl is None else f"{kl:.4f}"
            print(f"step {step + 1}: loss {loss_acc:.4f}  lr {lr:.2e}  {tps:,.0f} tok/s  {fmt(ppl, kl)}")
        elif (step + 1) % 10 == 0:
            print(f"step {step + 1}: loss {loss_acc:.4f}  lr {lr:.2e}  {tps:,.0f} tok/s")
        if step == 0:
            print(f"peak CUDA memory after step 1: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        log.write(f"{step + 1},{loss_acc:.6f},{lr:.6e},{tps:.1f},{ppl_s},{kl_s}\n")
        log.flush()

        if args.save_every and (step + 1) % args.save_every == 0 and not args.no_save:
            torch.save(hf.state_dict(), out / "model_state.pt")

    print(f"final: {fmt(*run_eval())}")
    if not args.no_save:
        torch.save(hf.state_dict(), out / "model_state.pt")
        print(f"saved to {out / 'model_state.pt'}")


if __name__ == "__main__":
    main()
