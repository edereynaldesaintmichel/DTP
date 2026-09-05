"""Streaming packed-token data for fine-tuning, and wikitext eval blocks."""

import torch


def packed_stream(tokenizer, seq_len, dataset_name, dataset_config=None, split="train",
                  text_key="text", shuffle_seed=1234, shuffle_buffer=10_000):
    """Yields token blocks of length seq_len + 1 (input = block[:-1], labels = block[1:])."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
    eos = tokenizer.eos_token_id
    buf = []
    for ex in ds:
        ids = tokenizer(ex[text_key], add_special_tokens=True)["input_ids"]
        ids.append(eos)
        buf.extend(ids)
        while len(buf) >= seq_len + 1:
            yield torch.tensor(buf[: seq_len + 1], dtype=torch.long)
            del buf[: seq_len + 1]


def batched(stream, batch_size):
    batch = []
    for blk in stream:
        batch.append(blk)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []


def wikitext_blocks(tokenizer, seq_len, n_blocks=64, split="test"):
    """[N, seq_len + 1] blocks from wikitext-2-raw-v1, BOS-prefixed if the
    tokenizer has a BOS token (Qwen's does not)."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    prefix = [] if bos is None else [bos]
    step = seq_len + 1 - len(prefix)
    blocks = []
    for i in range(0, len(ids) - step, step):
        blocks.append(torch.tensor(prefix + ids[i : i + step], dtype=torch.long))
        if len(blocks) >= n_blocks:
            break
    return torch.stack(blocks)


@torch.no_grad()
def perplexity(forward_fn, blocks, device, micro_batch=4):
    """forward_fn(input_ids) -> logits. Returns (ppl, mean_nll)."""
    import torch.nn.functional as F

    total_nll, total_tok = 0.0, 0
    for i in range(0, len(blocks), micro_batch):
        blk = blocks[i : i + micro_batch].to(device)
        logits = forward_fn(blk[:, :-1])
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            blk[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tok += blk[:, 1:].numel()
    mean = total_nll / total_tok
    return torch.tensor(mean).exp().item(), mean


def token_file_stream(path, seq_len, skip_blocks=0):
    """Blocks of seq_len + 1 tokens from a pre-tokenised file (scripts/pretokenize.py),
    same shape as packed_stream but bounded memory and fixed order."""
    import numpy as np

    toks = np.load(path, mmap_mode="r")
    step = seq_len + 1
    for i in range(skip_blocks * step, len(toks) - step + 1, step):
        yield torch.from_numpy(np.array(toks[i : i + step], dtype=np.int64))
