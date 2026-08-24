"""Generation smoke test for a DTP Qwen3 model.

  python scripts/generate.py --devices 4 --delta 4 \
      --checkpoint runs/dtp_l4_d4/model_state.pt \
      --prompt "The capital of France is"
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.model import load_dtp_qwen3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--delta", type=int, default=4)
    p.add_argument("--stage3-scale", default="sqrt_l", choices=["sqrt_l", "one"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = load_dtp_qwen3(
        args.model_id, n_devices=args.devices, delta=args.delta, dtype=dtype,
        device=device, state_dict_path=args.checkpoint, stage3_own_scale=args.stage3_scale,
    )
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, eos_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
