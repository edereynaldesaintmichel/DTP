"""Wikitext-2 perplexity: vanilla Qwen3 vs DTP at one or more delays.

Examples:
  python scripts/eval_ppl.py --devices 4 --deltas 0 1 2 4 8 --vanilla
  python scripts/eval_ppl.py --devices 4 --deltas 4 --checkpoint runs/dtp_l4_d4/model_state.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import perplexity, wikitext_blocks
from dtp.model import load_dtp_qwen3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--deltas", type=int, nargs="*", default=[0, 1, 2, 4, 8])
    p.add_argument("--checkpoint", default=None, help="state dict from finetune.py")
    p.add_argument("--vanilla", action="store_true", help="also eval the stock HF model")
    p.add_argument("--stage3-scale", default="sqrt_l", choices=["sqrt_l", "one"])
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--n-blocks", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=4)
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

    rows = []
    if args.vanilla:
        ppl, nll = perplexity(lambda x: dtp.hf(x).logits, blocks, device, args.micro_batch)
        rows.append(("vanilla", ppl, nll))
        print(f"vanilla                 ppl {ppl:10.3f}   nll {nll:.4f}")
    for d in args.deltas:
        dtp.delta = d
        assert d <= dtp.n_modules // 2
        ppl, nll = perplexity(lambda x: dtp(x).logits, blocks, device, args.micro_batch)
        rows.append((f"dtp L={args.devices} d={d}", ppl, nll))
        print(f"dtp L={args.devices} delta={d:<3d}      ppl {ppl:10.3f}   nll {nll:.4f}")

    print("\nname                          ppl        nll")
    for name, ppl, nll in rows:
        print(f"{name:<24}{ppl:12.3f}   {nll:.4f}")


if __name__ == "__main__":
    main()
