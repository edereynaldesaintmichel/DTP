"""Which neurons produce the massive activation after layer 2's MLP, on which
tokens, and which heads drive their pre-activation? Vanilla model, 2 wikitext
blocks. Light: one forward pass."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import wikitext_blocks

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
from transformers import AutoTokenizer, Qwen3ForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
hf = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.float32).cuda().eval()
blocks = wikitext_blocks(tok, 1024, 2)
ids = blocks[:, :-1].cuda()
layer = hf.model.layers[LAYER]
cap = {}
hs = [
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("act", i[0])),
    layer.self_attn.o_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("a", i[0])),
    layer.post_attention_layernorm.register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0])),
    hf.model.layers[LAYER + 1].input_layernorm.register_forward_pre_hook(lambda m, i: cap.__setitem__("x_next", i[0])),
]
with torch.no_grad():
    hf(ids)
for h in hs:
    h.remove()
B, S, I = cap["act"].shape
act = cap["act"].reshape(B * S, I)
Wd = layer.mlp.down_proj.weight  # [D, I]
contrib = act.abs() * Wd.norm(dim=0)[None]  # [T, I] residual-norm contribution per neuron per token
top = contrib.max(0).values.topk(6)
print(f"layer {LAYER}: max|residual| after MLP = {cap['x_next'].abs().max():.1f}; before = {cap['x'].abs().max():.1f}")
print("top neurons by max token contribution ||act_j * Wd_j||:")
for v, j in zip(top.values.tolist(), top.indices.tolist()):
    c = contrib[:, j]
    hot = (c > 0.1 * c.max()).nonzero().flatten()
    toks = [tok.decode([ids.flatten()[t].item()]) for t in hot[:12].tolist()]
    dim = Wd[:, j].abs().argmax().item()
    print(f"  neuron {j:5d}: max {v:8.1f}  mean {c.mean():7.2f}  fires on {len(hot)} tokens: {toks}  dominant out dim {dim}")

# which heads drive the top neuron's gate/up pre-activation on its hot tokens
j = top.indices[0].item()
H, hd = hf.config.num_attention_heads, layer.self_attn.head_dim
a = cap["a"].reshape(B * S, H, hd)
Wo = layer.self_attn.o_proj.weight.view(-1, H, hd)
x = cap["x"].reshape(B * S, -1)
rms = (x.pow(2).mean(-1) + hf.config.rms_norm_eps).sqrt()
gain = layer.post_attention_layernorm.weight
head_out = torch.einsum("the,dhe->htd", a, Wo)
Wg, Wu = layer.mlp.gate_proj.weight[j], layer.mlp.up_proj.weight[j]
pg = (head_out * gain / rms[None, :, None]) @ Wg  # [H, T]
pu = (head_out * gain / rms[None, :, None]) @ Wu
c = contrib[:, j]
hot = (c > 0.1 * c.max()).nonzero().flatten()
full_g = ((x * gain) / rms[:, None]) @ Wg
full_u = ((x * gain) / rms[:, None]) @ Wu
print(f"\nneuron {j} on its hot tokens: gate pre-act {full_g[hot].mean():.2f}, up pre-act {full_u[hot].mean():.2f}")
print("  per-head contribution to gate / up on hot tokens (KV group = head // 2):")
for h in range(H):
    print(f"   head {h:2d} (kv {h // 2}): gate {pg[h, hot].mean():+7.2f}   up {pu[h, hot].mean():+7.2f}")
print("  residual-only (everything but this layer's attention): gate "
      f"{(full_g[hot] - pg[:, hot].sum(0)).mean():+.2f}  up {(full_u[hot] - pu[:, hot].sum(0)).mean():+.2f}")
