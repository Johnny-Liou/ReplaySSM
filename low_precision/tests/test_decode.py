# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay through multiple low-precision checkpoint flushes."""

import importlib
import inspect

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 (SM103) required",
)


def make_decode(arch, precision):
    batch, slots, length = 2, 3, 16
    mamba = arch == "mamba2"
    heads, groups, rows = (32, 2, 64) if mamba else (16, 4, 128)

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(shape, device="cuda", dtype=dtype)

    state = zeros(slots, heads, rows, 128, dtype=torch.float16)
    low_shape = (slots, heads, rows, 128)
    if precision == "nvfp4":
        low_shape = (
            (slots, heads // 2, 2 * rows, 64) if mamba else (slots, heads, rows, 64)
        )
    low = zeros(*low_shape, dtype=torch.uint8)
    low_scale = torch.ones(slots, heads, rows, device="cuda")
    values = dict(
        state=state,
        state_lo=low,
        state_lo_scale=low_scale,
        sf_lo=zeros(slots, heads // 2 if mamba else heads, 1024, dtype=torch.uint8),
        sg_meta=zeros(slots, 3, groups),
        sg_parity=zeros(slots, groups, dtype=torch.int32),
        state_indices=torch.arange(1, batch + 1, device="cuda", dtype=torch.int32),
        is_flush=zeros(batch, dtype=torch.int8),
        out=zeros(batch, heads, rows, dtype=torch.bfloat16),
        stochastic_rounding_seed=1234,
        stochastic_rounding_salt=7,
    )
    values["sg_meta"][:, 2] = 1
    if mamba:
        values.update(
            x=torch.randn(batch, heads, rows, device="cuda", dtype=torch.bfloat16),
            dt=torch.full((batch, heads), -3.0, device="cuda"),
            A=-torch.ones(heads, device="cuda"),
            B=torch.randn(batch, groups, 128, device="cuda", dtype=torch.bfloat16),
            C=torch.randn(batch, groups, 128, device="cuda", dtype=torch.bfloat16),
            D=torch.ones(heads, rows, device="cuda"),
            dt_bias=zeros(heads),
            x_cache=zeros(slots, heads, length, rows, dtype=torch.bfloat16),
            dt_cache=zeros(slots, heads, length),
            B_cache=zeros(slots, groups, length, 128, dtype=torch.bfloat16),
        )
    else:
        values.update(
            q=torch.randn(batch, groups, 128, device="cuda", dtype=torch.bfloat16),
            k=torch.randn(batch, groups, 128, device="cuda", dtype=torch.bfloat16),
            v=torch.randn(batch, heads, rows, device="cuda", dtype=torch.bfloat16),
            a=torch.full((batch, heads), -3.0, device="cuda", dtype=torch.bfloat16),
            b=zeros(batch, heads, dtype=torch.bfloat16),
            a_log=zeros(heads),
            dt_bias=zeros(heads),
            scale=128**-0.5,
            u_cache=zeros(slots, heads, length, rows, dtype=torch.float16),
            v_cache=zeros(slots, heads, length, rows, dtype=torch.bfloat16),
            k_cache=zeros(slots, groups, length, 128, dtype=torch.bfloat16),
            g_cache=zeros(slots, heads, length),
            beta_cache=zeros(slots, heads, length),
        )
    position = zeros(batch, dtype=torch.int32)
    tag = torch.full((batch,), 16, device="cuda", dtype=torch.int64)
    names = (
        [precision]
        if mamba and precision == "fp16"
        else [precision + "_nonflush", precision + "_flush"]
    )
    calls = []
    for name in names:
        module = importlib.import_module(f"low_precision.{arch}.{name}")
        function = getattr(module, f"{arch}_{name}")
        kwargs = {
            key: value
            for key, value in values.items()
            if key in inspect.signature(function).parameters
        }
        kwargs["write_pos"] = (
            tag
            if name in ("fp16", "fp16_nonflush") or name.endswith("_flush")
            else position
        )
        calls.append((function, kwargs))

    def step():
        torch.remainder(tag, 16, out=position)
        torch.eq(position, 15, out=values["is_flush"])
        for function, kwargs in calls:
            function(**kwargs)
        tag.add_(1)

    return step, values, tag


@pytest.mark.parametrize("arch", ["mamba2", "gdn"])
@pytest.mark.parametrize("precision", ["fp16", "fp8", "nvfp4"])
def test_graph_preserves_state_and_rounding_events(arch, precision):
    torch.manual_seed(7)
    eager, expected, _ = make_decode(arch, precision)
    torch.manual_seed(7)
    captured, actual, tag = make_decode(arch, precision)
    # Compile every path and initialize FFI before capturing a fresh set of buffers.
    for _ in range(17):
        captured()
    torch.manual_seed(7)
    captured, actual, tag = make_decode(arch, precision)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured()
    for _ in range(33):
        eager()
        graph.replay()
    torch.cuda.synchronize()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert torch.isfinite(actual["out"]).all()
    assert tag.tolist() == [49, 49]
