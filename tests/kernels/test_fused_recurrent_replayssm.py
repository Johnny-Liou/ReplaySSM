# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_replayssm,
    fused_recurrent_gated_delta_rule_packed_decode,
)


def _make_mixed_qkv(
    batch: int,
    qkv_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    strided: bool,
) -> torch.Tensor:
    if strided:
        proj = torch.randn((batch, qkv_dim + 64), device=device, dtype=dtype)
        return proj[:, :qkv_dim]
    return torch.randn((batch, qkv_dim), device=device, dtype=dtype)


def _make_caches(
    num_state_slots: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    max_cache_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_cache = torch.zeros(
        (num_state_slots, num_v_heads, max_cache_len, head_v_dim),
        device=device,
        dtype=dtype,
    )
    k_cache = torch.zeros(
        (num_state_slots, num_q_heads, max_cache_len, head_k_dim),
        device=device,
        dtype=dtype,
    )
    g_cache = torch.zeros(
        (num_state_slots, num_v_heads, max_cache_len),
        device=device,
        dtype=torch.float32,
    )
    return d_cache, k_cache, g_cache


def _assert_output_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _assert_state_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    atol = 2e-2 if dtype != torch.float32 else 2e-3
    rtol = 1e-2 if dtype != torch.float32 else 1e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _packed_decode_step(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch = mixed_qkv.shape[0]
    num_v_heads = state.shape[1]
    head_v_dim = state.shape[2]
    out = torch.empty(
        (batch, 1, num_v_heads, head_v_dim),
        device=mixed_qkv.device,
        dtype=mixed_qkv.dtype,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return out


def _cached_decode_step(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ssm_state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch = mixed_qkv.shape[0]
    _, num_v_heads, head_v_dim, _ = state.shape
    out = torch.empty(
        (batch, 1, num_v_heads, head_v_dim),
        device=mixed_qkv.device,
        dtype=mixed_qkv.dtype,
    )
    fused_recurrent_gated_delta_rule_replayssm(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state,
        d_cache=caches[0],
        k_cache=caches[1],
        g_cache=caches[2],
        out=out,
        ssm_state_indices=ssm_state_indices,
        write_pos=write_pos,
        use_qk_l2norm_in_kernel=True,
    )
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
def test_fused_recurrent_replayssm_matches_packed_decode(
    dtype: torch.dtype,
    strided_mixed_qkv: bool,
):
    torch.manual_seed(0)

    batch = 8
    num_q_heads = 2
    num_v_heads = 4
    head_k_dim = 64
    head_v_dim = 64
    max_cache_len = 4
    qkv_dim = 2 * (num_q_heads * head_k_dim) + (num_v_heads * head_v_dim)
    num_state_slots = batch + 1
    scale = head_k_dim**-0.5
    device = torch.device("cuda")

    A_log = torch.randn((num_v_heads,), device=device, dtype=dtype)
    dt_bias = torch.randn((num_v_heads,), device=device, dtype=dtype)
    ssm_state_indices = torch.arange(
        1, batch + 1, device=device, dtype=torch.int32
    )
    ssm_state_indices[-2:] = -1

    state0 = torch.randn(
        (num_state_slots, num_v_heads, head_v_dim, head_k_dim),
        device=device,
        dtype=dtype,
    )
    state_packed = state0.clone()
    state_cached = state0.clone()
    caches = _make_caches(
        num_state_slots,
        num_q_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        max_cache_len,
        dtype,
        device,
    )

    for step in range(max_cache_len + 1):
        mixed_qkv = _make_mixed_qkv(
            batch, qkv_dim, dtype, device, strided_mixed_qkv
        )
        a = torch.randn((batch, num_v_heads), device=device, dtype=dtype)
        b = torch.randn((batch, num_v_heads), device=device, dtype=dtype)
        write_pos = torch.full(
            (batch,), step % max_cache_len, device=device, dtype=torch.int32
        )

        out_packed = _packed_decode_step(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            state_packed,
            ssm_state_indices,
            scale,
        )
        out_cached = _cached_decode_step(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            state_cached,
            caches,
            ssm_state_indices,
            write_pos,
            scale,
        )

        _assert_output_close(out_cached, out_packed, dtype)
        if step % max_cache_len == max_cache_len - 1:
            _assert_state_close(state_cached, state_packed, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_recurrent_replayssm_uses_per_row_write_pos(
    dtype: torch.dtype,
):
    torch.manual_seed(1)

    batch = 4
    num_q_heads = 2
    num_v_heads = 4
    head_k_dim = 64
    head_v_dim = 64
    max_cache_len = 4
    qkv_dim = 2 * (num_q_heads * head_k_dim) + (num_v_heads * head_v_dim)
    num_state_slots = 7
    scale = head_k_dim**-0.5
    device = torch.device("cuda")

    A_log = torch.randn((num_v_heads,), device=device, dtype=dtype)
    dt_bias = torch.randn((num_v_heads,), device=device, dtype=dtype)
    ssm_state_indices = torch.tensor([4, 2, 6, 1], device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)

    state0 = torch.randn(
        (num_state_slots, num_v_heads, head_v_dim, head_k_dim),
        device=device,
        dtype=dtype,
    )
    state_packed = state0.clone()
    state_cached = state0.clone()
    caches = _make_caches(
        num_state_slots,
        num_q_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        max_cache_len,
        dtype,
        device,
    )

    for row, prefix_len in enumerate(prefix_lens.tolist()):
        row_indices = ssm_state_indices[row : row + 1]
        for step in range(prefix_len):
            mixed_qkv = _make_mixed_qkv(1, qkv_dim, dtype, device, False)
            a = torch.randn((1, num_v_heads), device=device, dtype=dtype)
            b = torch.randn((1, num_v_heads), device=device, dtype=dtype)
            write_pos = torch.tensor([step], device=device, dtype=torch.int32)
            _packed_decode_step(
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                state_packed,
                row_indices,
                scale,
            )
            _cached_decode_step(
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                state_cached,
                caches,
                row_indices,
                write_pos,
                scale,
            )

    mixed_qkv = _make_mixed_qkv(batch, qkv_dim, dtype, device, False)
    a = torch.randn((batch, num_v_heads), device=device, dtype=dtype)
    b = torch.randn((batch, num_v_heads), device=device, dtype=dtype)

    out_packed = _packed_decode_step(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_packed,
        ssm_state_indices,
        scale,
    )
    out_cached = _cached_decode_step(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_cached,
        caches,
        ssm_state_indices,
        prefix_lens,
        scale,
    )

    _assert_output_close(out_cached, out_packed, dtype)
