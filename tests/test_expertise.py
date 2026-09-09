"""Layout optimiser variants (scripts/expertise.py) and the seeded token stream."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtp.data import token_file_stream
from scripts.expertise import Chain, random_partition

NL, G, I, L = 5, 8, 64, 4


def toy_chain(sign=1, seed=0):
    gen = torch.Generator().manual_seed(seed)
    S_up = [torch.rand(G, I, generator=gen, dtype=torch.float64) for _ in range(NL)]
    S_dn = [torch.rand(I, G, generator=gen, dtype=torch.float64) for _ in range(NL - 1)]
    S_up = [s / s.sum() for s in S_up]
    S_dn = [s / s.sum() for s in S_dn]
    return Chain(S_up, S_dn, L, sign=sign)


def identity():
    return [torch.arange(G) // (G // L) for _ in range(NL)], [torch.arange(I) // (I // L) for _ in range(NL)]


def balanced(dev, n):
    return torch.bincount(dev, minlength=L).tolist() == [n // L] * L


def test_maximise_beats_identity_and_random():
    c = toy_chain()
    h0, n0 = identity()
    h, n, obj = c.solve(h0, n0, verbose=False)
    assert abs(obj - c.objective(h, n)) < 1e-9
    assert obj > c.objective(h0, n0)
    gen = torch.Generator().manual_seed(1)
    parts = [random_partition(G, I, L, gen) for _ in range(NL)]
    assert obj > c.objective([p[0] for p in parts], [p[1] for p in parts])
    assert all(balanced(x, G) for x in h) and all(balanced(x, I) for x in n)


def test_minimise_goes_below_identity():
    c = toy_chain(sign=-1)
    h0, n0 = identity()
    h, n, obj = c.solve(h0, n0, verbose=False)
    assert abs(obj - c.objective(h, n)) < 1e-9
    assert obj < c.objective(h0, n0)
    assert all(balanced(x, G) for x in h) and all(balanced(x, I) for x in n)


def test_frozen_layers_and_partial_freedom():
    c = toy_chain()
    h0, n0 = identity()
    free = [i in (1, 3) for i in range(NL)]
    h, n, _ = c.solve(h0, n0, free_heads=free, free_neurons=free, verbose=False)
    for i in range(NL):
        if not free[i]:
            assert torch.equal(h[i], h0[i]) and torch.equal(n[i], n0[i])
    assert any(not torch.equal(n[i], n0[i]) for i in (1, 3))
    h, n, _ = c.solve(h0, n0, free_heads=[False] * NL, verbose=False)  # neurons only
    assert all(torch.equal(h[i], h0[i]) for i in range(NL))
    h, n, _ = c.solve(h0, n0, free_neurons=[False] * NL, verbose=False)  # heads only
    assert all(torch.equal(n[i], n0[i]) for i in range(NL))


def test_token_file_stream_shuffle(tmp_path):
    seq_len, n_blocks = 7, 20
    toks = np.arange(n_blocks * (seq_len + 1), dtype=np.uint16)
    np.save(tmp_path / "t.npy", toks)
    fixed = [b[0].item() for b in token_file_stream(tmp_path / "t.npy", seq_len, skip_blocks=3)]
    assert fixed == [b * (seq_len + 1) for b in range(3, n_blocks)]
    s1 = [b[0].item() for b in token_file_stream(tmp_path / "t.npy", seq_len, 3, shuffle_seed=1)]
    s1b = [b[0].item() for b in token_file_stream(tmp_path / "t.npy", seq_len, 3, shuffle_seed=1)]
    s2 = [b[0].item() for b in token_file_stream(tmp_path / "t.npy", seq_len, 3, shuffle_seed=2)]
    assert sorted(s1) == fixed and s1 == s1b and s1 != s2 and s1 != fixed
    blk = next(token_file_stream(tmp_path / "t.npy", seq_len, 0, shuffle_seed=1))
    assert blk.shape == (seq_len + 1,) and blk.dtype == torch.int64
