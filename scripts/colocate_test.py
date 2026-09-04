"""Causal test: does co-locating layer 2's massive-activation neurons with
their driver KV group (3) change untrained DTP loss at delta=1, holding
everything else fixed? Start from the bad and good random layouts."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import perplexity, wikitext_blocks
from dtp.model import DTPQwen3
from scripts.expertise import apply_permutation, random_partition

MASSIVE = [55, 128, 1489, 321, 46, 646]
GROUP = 3
LAYER = 2
from transformers import AutoTokenizer, Qwen3ForCausalLM

device, L = "cuda", 4
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
blocks = wikitext_blocks(tok, 1024, 32)
base_sd = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.float32).state_dict()
hf = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.float32).to(device).eval()
dtp = DTPQwen3(hf, n_devices=L, delta=1).to(device).eval()
KV, I, NL = 8, 3072, 28


def parts_for(seed):
    gen = torch.Generator().manual_seed(1000 + seed)
    ps = [random_partition(KV, I, L, gen) for _ in range(NL)]
    return [h for h, _ in ps], [n for _, n in ps]


def ev(h, n):
    hf.load_state_dict(base_sd)
    apply_permutation(hf, h, n, L)
    return perplexity(lambda x: dtp(x).logits, blocks, device, 4)[1]


def with_massive_on(nd, dev, gen):
    """Move the massive neurons of layer LAYER to device `dev`, swapping with
    random non-massive neurons currently on `dev` to keep the balance."""
    n = nd.clone()
    for j in MASSIVE:
        if n[j] == dev:
            continue
        pool = ((n == dev) & ~torch.isin(torch.arange(I), torch.tensor(MASSIVE))).nonzero().flatten()
        k = pool[torch.randint(len(pool), (1,), generator=gen)].item()
        n[k], n[j] = n[j].clone(), n[k].clone()
    return n


gen = torch.Generator().manual_seed(7)
for name, seed in (("bad", 0), ("good", 3)):
    h, n = parts_for(seed)
    g_dev = h[LAYER][GROUP].item()
    base = ev(h, n)
    print(f"{name} (seed {seed}): nll {base:.4f}; layer-2 group 3 on device {g_dev}, massive on {[n[LAYER][j].item() for j in MASSIVE]}")
    n2 = list(n); n2[LAYER] = with_massive_on(n[LAYER], g_dev, gen)
    print(f"  massive neurons moved onto group 3's device:   nll {ev(h, n2):.4f}")
    other = (g_dev + 1) % L
    n3 = list(n); n3[LAYER] = with_massive_on(n[LAYER], other, gen)
    print(f"  massive neurons all on another device ({other}):  nll {ev(h, n3):.4f}")
    print(flush=True)

# max|residual| after layer 2 under DTP for the two bad variants, to see whether the circuit fires
h, n = parts_for(0)
g_dev = h[LAYER][GROUP].item()
for tag, nn_ in (("bad", n), ("bad, massive co-located", [n[i] if i != LAYER else with_massive_on(n[LAYER], g_dev, gen) for i in range(NL)])):
    hf.load_state_dict(base_sd)
    apply_permutation(hf, h, nn_, L)
    ids = blocks[:2, :-1].to(device)
    model = hf.model
    with torch.no_grad():
        x = model.embed_tokens(ids).unsqueeze(0).expand(L, *ids.shape, -1)
        pos = torch.arange(ids.shape[1], device=device)[None].expand(ids.shape[0], -1)
        pe = model.rotary_emb(model.embed_tokens(ids), pos)
        queue, nmod = [], 0
        for li, layer in enumerate(model.layers[:4]):
            for kind in ("attn", "mlp"):
                kw = dict(pos_emb=pe, mask=None, is_causal=True) if kind == "attn" else {}
                o = dtp._module_out(layer, kind, x, **kw)
                x = x + dtp._own_scale(nmod) * o
                queue.append(o)
                if nmod >= 1:
                    op = queue[nmod - 1]
                    x = x + (op.sum(0, keepdim=True) - op)
                if nmod == 5:
                    print(f"{tag}: after layer-2 MLP, per-device max|x| = {[round(v, 1) for v in x.abs().amax((1, 2, 3)).tolist()]}  (vanilla 7020)")
                nmod += 1
