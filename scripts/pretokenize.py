"""Pre-tokenise a fixed slice of FineWeb-Edu into a .npy token file.

The streaming reader (dtp.data.packed_stream) prefetches whole ~2 GB parquet
shards into RAM, which overflows small containers. This reads one shard row
group by row group instead, packs documents as BOS + tokens + EOS, and stops at
--tokens. affinity_stats.py and finetune.py consume the file via --token-file.

    python scripts/pretokenize.py --model-id h2oai/h2o-danube3-500m-base --tokens 34_000_000 --out runs/fineweb_tokens.npy
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--shard", default="sample/10BT/000_00000.parquet")
    p.add_argument("--tokens", type=int, default=34_000_000)
    p.add_argument("--out", default="runs/fineweb_tokens.npy")
    p.add_argument("--keep-shard", action="store_true")
    args = p.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    dtype = np.uint16 if len(tok) < 2**16 else np.uint32
    eos = tok.eos_token_id
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(args.repo, args.shard, repo_type="dataset", local_dir=out.parent / "shard")
    pf = pq.ParquetFile(path)
    chunks, n = [], 0
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        for ids in tok(texts, add_special_tokens=True)["input_ids"]:
            ids.append(eos)
            chunks.append(np.asarray(ids, dtype=dtype))
            n += len(ids)
        print(f"row group {rg}: {n:,} tokens", flush=True)
        if n >= args.tokens:
            break
    toks = np.concatenate(chunks)[: args.tokens]
    np.save(out, toks)
    print(f"saved {out}: {len(toks):,} tokens, dtype {toks.dtype}")
    if not args.keep_shard:
        import shutil

        shutil.rmtree(out.parent / "shard", ignore_errors=True)


if __name__ == "__main__":
    main()
