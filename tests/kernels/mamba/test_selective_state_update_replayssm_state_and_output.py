# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.mamba.utils import (
    allocate_update_caches,
    selective_state_update_replayssm_state_and_output_ref,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_state_and_output import (
    selective_state_update_replayssm_state_and_output,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-3, 1e-2
    return 6e-2, 2e-1


def _tied_A(nheads: int, headdim: int, dstate: int, device: str) -> torch.Tensor:
    A = -torch.rand(nheads, device=device) - 1.0
    return A.view(nheads, 1, 1).expand(nheads, headdim, dstate)


def _tied_dt(
    batch: int,
    nheads: int,
    headdim: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    dt = torch.randn(batch, nheads, device=device, dtype=dtype)
    return dt.unsqueeze(-1).expand(batch, nheads, headdim)


def _tied_dt_bias(nheads: int, headdim: int, device: str) -> torch.Tensor:
    dt_bias = torch.rand(nheads, device=device) - 4.0
    return dt_bias.view(nheads, 1).expand(nheads, headdim)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("ngroups", [1, 4])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("max_cache_len", [1, 4])
def test_selective_state_update_replayssm_state_and_output_matches_baseline_decode(
    max_cache_len: int,
    dstate: int,
    ngroups: int,
    has_z: bool,
    itype: torch.dtype,
):
    device = "cuda"
    rtol, atol = _tolerances(itype)
    set_random_seed(0)

    batch = 2
    nheads = 4
    headdim = 64
    num_steps = 2 * max_cache_len

    state = torch.randn(batch, nheads, headdim, dstate, dtype=itype, device=device)
    state_baseline = state.clone()
    state_cached = state.clone()
    state_ref = state.clone()

    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device)
    D = torch.randn(nheads, headdim, device=device)

    x_cache, dt_cache, B_cache, write_pos = allocate_update_caches(
        batch, nheads, ngroups, headdim, dstate, max_cache_len, state.device, itype,
        itype)
    x_cache_ref, dt_cache_ref, B_cache_ref, write_pos_ref = allocate_update_caches(
        batch, nheads, ngroups, headdim, dstate, max_cache_len, state.device, itype,
        itype)

    for _ in range(num_steps):
        x = torch.randn(batch, nheads, headdim, device=device, dtype=itype)
        dt = _tied_dt(batch, nheads, headdim, device, itype)
        B = torch.randn(batch, ngroups, dstate, device=device, dtype=itype)
        C = torch.randn(batch, ngroups, dstate, device=device, dtype=itype)
        z = torch.randn_like(x) if has_z else None

        out_baseline = torch.empty_like(x)
        selective_state_update(
            state_baseline,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=out_baseline,
        )

        out_cached = torch.empty_like(x)
        is_flush = write_pos == max_cache_len - 1
        selective_state_update_replayssm_state_and_output(
            state_cached,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            out=out_cached,
        )

        out_ref = selective_state_update_replayssm_state_and_output_ref(
            state_ref,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=x_cache_ref,
            dt_cache=dt_cache_ref,
            B_cache=B_cache_ref,
            write_pos=write_pos_ref,
            max_cache_len=max_cache_len,
        )

        torch.testing.assert_close(out_cached, out_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(out_cached, out_baseline, rtol=rtol, atol=atol)
        torch.testing.assert_close(x_cache, x_cache_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(dt_cache, dt_cache_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(B_cache, B_cache_ref, rtol=rtol, atol=atol)

        if bool(is_flush.all()):
            torch.testing.assert_close(
                state_cached, state_baseline, rtol=rtol, atol=atol)
            torch.testing.assert_close(
                state_ref, state_baseline, rtol=rtol, atol=atol)

        next_write_pos = torch.where(
            is_flush, torch.zeros_like(write_pos), write_pos + 1)
        write_pos.copy_(next_write_pos)
        write_pos_ref.copy_(next_write_pos)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_padding", [False, True])
def test_selective_state_update_replayssm_state_and_output_with_batch_indices(
    with_padding: bool,
    itype: torch.dtype,
):
    device = "cuda"
    rtol, atol = _tolerances(itype)
    set_random_seed(0)

    batch = 3
    padding = 2 if with_padding else 0
    padded_batch = batch + padding
    total_state_slots = 16
    nheads = 4
    ngroups = 2
    headdim = 64
    dstate = 16
    max_cache_len = 4
    num_steps = 2 * max_cache_len

    state = torch.randn(
        total_state_slots, nheads, headdim, dstate, dtype=itype, device=device)
    state_baseline = state.clone()
    state_cached = state.clone()
    state_before = state.clone()

    state_indices = (
        torch.randperm(total_state_slots - 1, device=device)[:batch] + 1
    ).to(torch.int32)
    state_batch_indices = torch.cat([
        state_indices,
        torch.full((padding,), NULL_BLOCK_ID, dtype=torch.int32, device=device),
    ])
    unused_states = torch.ones(total_state_slots, dtype=torch.bool, device=device)
    unused_states[state_indices] = False

    A = _tied_A(nheads, headdim, dstate, device)
    dt_bias = _tied_dt_bias(nheads, headdim, device)
    D = torch.randn(nheads, headdim, device=device)
    x_cache = torch.zeros(
        total_state_slots, nheads, max_cache_len, headdim, device=device, dtype=itype)
    dt_cache = torch.zeros(
        total_state_slots, nheads, max_cache_len, device=device, dtype=torch.float32)
    B_cache = torch.zeros(
        total_state_slots, ngroups, max_cache_len, dstate, device=device, dtype=itype)
    write_pos = torch.zeros(padded_batch, dtype=torch.int32, device=device)

    for _ in range(num_steps):
        x = torch.randn(padded_batch, nheads, headdim, device=device, dtype=itype)
        dt = _tied_dt(padded_batch, nheads, headdim, device, itype)
        B = torch.randn(padded_batch, ngroups, dstate, device=device, dtype=itype)
        C = torch.randn(padded_batch, ngroups, dstate, device=device, dtype=itype)
        z = torch.randn_like(x)

        out_baseline = torch.empty_like(x)
        selective_state_update(
            state_baseline,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            out=out_baseline,
        )

        out_cached = torch.full_like(x, 42)
        is_flush = write_pos == max_cache_len - 1
        selective_state_update_replayssm_state_and_output(
            state_cached,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            write_pos=write_pos,
            is_flush=is_flush,
            max_cache_len=max_cache_len,
            state_batch_indices=state_batch_indices,
            out=out_cached,
        )

        torch.testing.assert_close(
            out_cached[:batch], out_baseline[:batch], rtol=rtol, atol=atol)
        if with_padding:
            assert torch.equal(out_cached[batch:], torch.full_like(
                out_cached[batch:], 42))

        if bool(is_flush[:batch].all()):
            torch.testing.assert_close(
                state_cached[state_indices],
                state_baseline[state_indices],
                rtol=rtol,
                atol=atol,
            )

        next_write_pos = torch.where(
            is_flush, torch.zeros_like(write_pos), write_pos + 1)
        write_pos.copy_(next_write_pos)

    assert torch.equal(state_cached[unused_states], state_before[unused_states])
    assert torch.equal(state_baseline[unused_states], state_before[unused_states])
