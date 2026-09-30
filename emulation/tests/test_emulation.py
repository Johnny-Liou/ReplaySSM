# SPDX-License-Identifier: Apache-2.0
"""Checkpoint independence, window progression and NVFP4 underflow behavior."""

import pytest
import torch
import triton
import triton.language as tl
from emulation import GDNReplay, Mamba2Replay
from emulation.quant_tl import quantize_nvfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_case(arch, mode="tp4_hi16_sr"):
    state = torch.randn(2, 4, 128, 128, device="cuda") * 0.1
    if arch == "mamba2":
        runner = Mamba2Replay(state, 2, mode)
        args = (
            torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16),
            torch.full((2, 4), -3.0, device="cuda"),
            torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16),
            -torch.ones(4, device="cuda"),
            torch.ones(4, device="cuda"),
            torch.zeros(4, device="cuda"),
        )
    else:
        runner = GDNReplay(state, 2, mode)
        args = (
            torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16),
            torch.full((2, 4), -3.0, device="cuda"),
            torch.zeros(2, 4, device="cuda"),
            torch.zeros(4, device="cuda"),
            torch.zeros(4, device="cuda"),
        )
    return runner, args


@pytest.mark.parametrize("arch", ["mamba2", "gdn"])
def test_read_copy_does_not_inherit_anchor_rounding(arch):
    torch.manual_seed(9)
    rounded, args = make_case(arch)
    cls = type(rounded)
    exact = cls(rounded.state_hi, 2, "tp4_hi32")
    rounded.step(*args)
    exact.step(*args)
    assert not torch.equal(rounded.state_hi, exact.state_hi)
    torch.testing.assert_close(rounded.state_lo, exact.state_lo, atol=0, rtol=0)


@pytest.mark.parametrize("arch", ["mamba2", "gdn"])
def test_cuda_graph_advances_flush_and_random_events(arch):
    torch.manual_seed(11)
    warmup, args = make_case(arch)
    warmup.step(*args)
    # Compile the flush path and initialize CUDA before capture.
    for _ in range(16):
        warmup.step(*args)
    eager = type(warmup)(warmup.state_hi, 2)
    captured = type(warmup)(warmup.state_hi, 2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured.step(*args)
    for _ in range(49):
        expected = eager.step(*args)
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.output, expected, atol=0, rtol=0)
    torch.testing.assert_close(captured.state_hi, eager.state_hi, atol=0, rtol=0)
    torch.testing.assert_close(captured.state_lo, eager.state_lo, atol=0, rtol=0)
    assert captured.steps.tolist() == [49, 49]
    assert captured.window_idx.tolist() == [3, 3]


@triton.jit
def _tiny_nvfp4(x, out):
    i = tl.arange(0, 128)
    values = tl.load(x + i)[None, :]
    # A large outer scale makes this block's E4M3 scale round to zero.
    rounded = quantize_nvfp4(
        values, tl.full((1, 128), 0.999, tl.float32), 1.0, True, 8, 16
    )
    tl.store(out + i, tl.reshape(rounded, (128,)))


@pytest.mark.parametrize("value", [0.0, 1e-8])
def test_nvfp4_zero_scale_blocks_stay_zero_under_sr(value):
    x = torch.full((128,), value, device="cuda")
    out = torch.empty_like(x)
    _tiny_nvfp4[(1,)](x, out)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) == 0
