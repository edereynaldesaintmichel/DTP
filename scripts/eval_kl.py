"""Per-token KL(vanilla || DTP) and top-1 agreement on wikitext-2, per delta.

Measures how far the DTP model's next-token distribution drifts from the
vanilla model's, which is more sensitive than perplexity alone (a model can
have decent ppl while disagreeing with the original on many tokens).

  python scripts/eval_kl.py --devices 4 --deltas 0 1 2 4 8
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import wikitext_blocks
from dtp.model import load_dtp_qwen3


@torch.no_grad()
def kl_stats(ref_logits, logits):
    """Sum of per-token KL(ref || model), top-1 agreements, token count."""
    logp = F.log_softmax(ref_logits.float(), dim=-1)
    logq = F.log_softmax(logits.float(), dim=-1)
    kl = (logp.exp() * (logp - logq)).sum(-1)
    agree = (ref_logits.argmax(-1) == logits.argmax(-1))
    return kl.sum().item(), agree.sum().item(), kl.numel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--deltas", type=int, nargs="*", default=[0, 1, 2, 4, 8])
    p.add_argument("--checkpoint", default=None, help="state dict from finetune.py")
    p.add_argument("--stage3-scale", default="sqrt_l", choices=["sqrt_l", "one"])
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--n-blocks", type=int, default=32)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    blocks = wikitext_blocks(tok, args.seq_len, args.n_blocks)
    print(f"device={device} dtype={args.dtype} blocks={len(blocks)}x{args.seq_len}")

    dtp = load_dtp_qwen3(
        args.model_id, n_devices=args.devices, delta=0, dtype=dtype, device=device,
        state_dict_path=args.checkpoint, stage3_own_scale=args.stage3_scale,
    )
    dtp.eval()

    stats = {d: [0.0, 0, 0] for d in args.deltas}  # kl_sum, agree, n_tok
    with torch.no_grad():
        for i in range(0, len(blocks), args.micro_batch):
            x = blocks[i : i + args.micro_batch, :-1].to(device)
            ref = dtp.hf(x).logits
            for d in args.deltas:
                dtp.delta = d
                assert d <= dtp.n_modules // 2
                kl, agree, n = kl_stats(ref, dtp(x).logits)
                stats[d][0] += kl
                stats[d][1] += agree
                stats[d][2] += n
            del ref
            print(f"  {min(i + args.micro_batch, len(blocks))}/{len(blocks)} blocks", flush=True)

    print(f"\nKL(vanilla || dtp) per token, L={args.devices}")
    print("delta      mean KL (nats)   top-1 agree")
    for d in args.deltas:
        kl, agree, n = stats[d]
        print(f"{d:<8d}{kl / n:>14.4f}{agree / n:>14.1%}")


if __name__ == "__main__":
    main()
