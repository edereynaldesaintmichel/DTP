import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function


class A2AStart(Function):

    @staticmethod
    def forward(ctx, x, send_splits, recv_splits, group, state, key):
        out = x.new_empty((sum(recv_splits),) + tuple(x.shape[1:]))
        work = dist.all_to_all_single(
            out, x.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=group, async_op=True,
        )
        state[key + "/fwd"] = work
        ctx.state, ctx.key = state, key
        return out

    @staticmethod
    def backward(ctx, _grad_buf_ignored):
        work, grad_x = ctx.state.pop(ctx.key + "/bwd")
        work.wait()
        return grad_x, None, None, None, None, None


class A2AFinish(Function):

    @staticmethod
    def forward(ctx, buf, send_splits, recv_splits, group, state, key):
        work = state.pop(key + "/fwd", None)
        if work is not None:
            work.wait()
        ctx.state, ctx.key = state, key
        ctx.send_splits, ctx.recv_splits, ctx.group = send_splits, recv_splits, group
        return buf.view_as(buf)

    @staticmethod
    def backward(ctx, grad_y):
        grad_x = grad_y.new_empty(
            (sum(ctx.send_splits),) + tuple(grad_y.shape[1:]))
        work = dist.all_to_all_single(
            grad_x, grad_y.contiguous(),
            output_split_sizes=ctx.send_splits,
            input_split_sizes=ctx.recv_splits,
            group=ctx.group, async_op=True,
        )
        ctx.state[ctx.key + "/bwd"] = (work, grad_x)
        return grad_y, None, None, None, None, None


def exchange_split_sizes(send_counts, group):
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    return send_counts.tolist(), recv_counts.tolist()


def exchange_int_rows(x, send_splits, recv_splits, group):
    out = x.new_empty(sum(recv_splits))
    dist.all_to_all_single(out, x, output_split_sizes=recv_splits,
                           input_split_sizes=send_splits, group=group)
    return out


@dataclass
class RoutingMeta:
    order: torch.Tensor
    src_token: torch.Tensor
    gate: torch.Tensor
    send_splits: list
    recv_splits: list
    recv_local_expert: torch.Tensor
    n_tokens: int


@dataclass
class Carry:
    attn_in: torch.Tensor
    pending: Optional[tuple]
    state: dict


class FarSkipMoELayer(nn.Module):
    def __init__(self, idx, d, n_heads, n_experts_global, top_k, d_ff, group):
        super().__init__()
        self.idx, self.d, self.n_heads, self.top_k = idx, d, n_heads, top_k
        self.group = group
        world = dist.get_world_size(group)
        assert n_experts_global % world == 0
        self.experts_per_rank = n_experts_global // world
        self.n_experts_global = n_experts_global

        self.norm_attn = nn.LayerNorm(d)
        self.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.norm_mlp = nn.LayerNorm(d)
        self.router = nn.Linear(d, n_experts_global, bias=False)

        def mlp():
            return nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.GELU(),
                                 nn.Linear(d_ff, d, bias=False))
        self.local_experts = nn.ModuleList(mlp() for _ in range(self.experts_per_rank))
        self.shared_expert = mlp()

    def route(self, h):
        T = h.shape[0]
        world = dist.get_world_size(self.group)
        probs = self.router(h).softmax(-1)
        gate, expert = probs.topk(self.top_k, dim=-1)
        gate = gate / gate.sum(-1, keepdim=True)

        flat_expert = expert.reshape(-1)
        src_token = torch.arange(T, device=h.device).repeat_interleave(self.top_k)
        order = torch.argsort(flat_expert, stable=True)
        permuted = h[src_token[order]]

        dest_rank = flat_expert[order] // self.experts_per_rank
        send_counts = torch.bincount(dest_rank, minlength=world)
        send_splits, recv_splits = exchange_split_sizes(send_counts, self.group)
        recv_local_expert = exchange_int_rows(
            flat_expert[order] % self.experts_per_rank,
            send_splits, recv_splits, self.group)

        meta = RoutingMeta(order, src_token, gate.reshape(-1)[order],
                           send_splits, recv_splits, recv_local_expert, T)
        return permuted, meta

    def run_local_experts(self, x, meta):
        ids = meta.recv_local_expert
        order = torch.argsort(ids, stable=True)
        xs = x[order]
        counts = torch.bincount(ids, minlength=self.experts_per_rank).tolist()
        outs, start = [], 0
        for expert, c in zip(self.local_experts, counts):
            outs.append(expert(xs[start:start + c]))
            start += c
        ys = torch.cat(outs) if outs else xs
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        return ys[inv]

    def unmix(self, combined, meta):
        weighted = combined * meta.gate.unsqueeze(1)
        zeros = combined.new_zeros(meta.n_tokens, self.d)
        return zeros.index_add(0, meta.src_token[meta.order], weighted)

    def forward(self, carry: Carry) -> Carry:
        S = carry.state

        ha = self.norm_attn(carry.attn_in)
        T = ha.shape[0]
        q, k, v = self.qkv_proj(ha).chunk(3, dim=-1)

        mlp_in = carry.attn_in
        if carry.pending is not None:
            comb_buf, prev_meta, prev_key = carry.pending
            combined = A2AFinish.apply(
                comb_buf,
                prev_meta.recv_splits, prev_meta.send_splits,
                self.group, S, prev_key)
            mlp_in = mlp_in + self.unmix(combined, prev_meta)

        hm = self.norm_mlp(mlp_in)
        permuted, meta = self.route(hm)

        key_d = f"L{self.idx}/dispatch"
        disp_buf = A2AStart.apply(permuted, meta.send_splits, meta.recv_splits,
                                  self.group, S, key_d)

        def heads(t):
            return t.view(T, self.n_heads, -1).transpose(0, 1).unsqueeze(0)
        attn = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        attn_out = self.o_proj(attn.squeeze(0).transpose(0, 1).reshape(T, self.d))

        recv = A2AFinish.apply(disp_buf, meta.send_splits, meta.recv_splits,
                               self.group, S, key_d)
        expert_out = self.run_local_experts(recv, meta)

        key_c = f"L{self.idx}/combine"
        comb_buf = A2AStart.apply(expert_out, meta.recv_splits, meta.send_splits,
                                  self.group, S, key_c)

        shared_out = self.shared_expert(hm)

        return Carry(attn_in=mlp_in + attn_out + shared_out,
                     pending=(comb_buf, meta, key_c),
                     state=S)


class FarSkipStack(nn.Module):
    def __init__(self, n_layers, d, n_heads, n_experts_global, top_k, d_ff, group):
        super().__init__()
        self.layers = nn.ModuleList(
            FarSkipMoELayer(i, d, n_heads, n_experts_global, top_k, d_ff, group)
            for i in range(n_layers))
        self.final_norm = nn.LayerNorm(d)
        self.group = group

    def forward(self, x):
        carry = Carry(attn_in=x, pending=None, state={})
        for layer in self.layers:
            carry = layer(carry)
        out = carry.attn_in
        if carry.pending is not None:
            comb_buf, meta, key = carry.pending
            last_layer = self.layers[-1]
            combined = A2AFinish.apply(comb_buf, meta.recv_splits, meta.send_splits,
                                       self.group, carry.state, key)
            out = out + last_layer.unmix(combined, meta)
        return self.final_norm(out)


def main():
    if not dist.is_initialized():
        if "RANK" in os.environ:
            dist.init_process_group(backend="gloo")
        else:
            dist.init_process_group(backend="gloo",
                                    init_method="tcp://127.0.0.1:29511",
                                    rank=0, world_size=1)
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)

    model = FarSkipStack(n_layers=3, d=64, n_heads=4,
                         n_experts_global=4 * world, top_k=2, d_ff=128,
                         group=group)

    x = torch.randn(32, 64, requires_grad=True)
    target = torch.randn(32, 64)
    y = model(x)
    loss = F.mse_loss(y, target)
    loss.backward()

    router_grad = model.layers[0].router.weight.grad.norm().item()
    expert_grad = sum(p.grad.norm().item()
                      for e in model.layers[0].local_experts
                      for p in e.parameters())
    print(f"[rank {rank}/{world}] loss={loss.item():.4f} "
          f"|grad x|={x.grad.norm().item():.4f} "
          f"|grad router|={router_grad:.4f} |grad experts|={expert_grad:.4f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()