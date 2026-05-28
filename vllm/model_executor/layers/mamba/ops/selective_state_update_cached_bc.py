# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from FlashMamba's cached-bc selective_state_update_cached.py.

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _cached_bc_precompute_kernel(
    B_ptr,
    C_ptr,
    B_cache_ptr,
    write_pos_ptr,
    is_flush_ptr,
    BC_pre_ptr,
    state_batch_indices_ptr,
    null_block_id,
    # Matrix dimensions
    batch,
    ngroups,
    dstate,
    # Input strides
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    # Cache strides
    stride_Bc_batch,
    stride_Bc_group,
    stride_Bc_pos,
    stride_Bc_dstate,
    stride_BC_batch,
    stride_BC_group,
    stride_BC_pos,
    stride_state_indices_batch,
    stride_state_indices_T,
    # Meta-parameters
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # heuristic-computed
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)

    # On flush steps the main kernel does not read BC_pre, so skip the work.
    is_flush = tl.load(is_flush_ptr + pid_b) != 0
    if is_flush:
        return

    if HAS_STATE_BATCH_INDICES:
        state_batch_idx = tl.load(
            state_batch_indices_ptr
            + pid_b * stride_state_indices_batch
            + 0 * stride_state_indices_T
        ).to(tl.int64)
        if state_batch_idx == null_block_id:
            return
    else:
        state_batch_idx = pid_b

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    write_pos = tl.load(write_pos_ptr + pid_b).to(tl.int64)

    B_ptr += pid_b * stride_B_batch + pid_g * stride_B_group
    C_ptr += pid_b * stride_C_batch + pid_g * stride_C_group
    B_cache_ptr += (
        state_batch_idx * stride_Bc_batch + pid_g * stride_Bc_group
    )
    BC_pre_ptr += pid_b * stride_BC_batch + pid_g * stride_BC_group

    B_cur = tl.load(
        B_ptr + offs_n * stride_B_dstate,
        mask=offs_n < dstate,
        other=0.0,
    )
    C = tl.load(
        C_ptr + offs_n * stride_C_dstate,
        mask=offs_n < dstate,
        other=0.0,
    )
    B_cache_ptrs = (
        B_cache_ptr
        + offs_k[:, None] * stride_Bc_pos
        + offs_n[None, :] * stride_Bc_dstate
    )
    B_cache = tl.load(
        B_cache_ptrs,
        mask=(offs_k[:, None] < write_pos) & (offs_n[None, :] < dstate),
        other=0.0,
    )
    B_all = tl.where(offs_k[:, None] == write_pos, B_cur[None, :], B_cache)
    BC = tl.sum(B_all.to(tl.float32) * C[None, :].to(tl.float32), axis=1)

    tl.store(
        BC_pre_ptr + offs_k * stride_BC_pos,
        BC,
        mask=(offs_k <= write_pos) & (offs_k < MAX_CACHE_LEN),
    )


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _selective_scan_update_cached_bc_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    BC_pre_ptr,
    write_pos_ptr,
    is_flush_ptr,
    state_batch_indices_ptr,
    null_block_id,
    # Matrix dimensions
    batch,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # State strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # Input strides
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    # Cache strides
    stride_xc_batch,
    stride_xc_head,
    stride_xc_dim,
    stride_xc_pos,
    stride_dtc_batch,
    stride_dtc_head,
    stride_dtc_pos,
    stride_Bc_batch,
    stride_Bc_group,
    stride_Bc_pos,
    stride_Bc_dstate,
    stride_BC_batch,
    stride_BC_group,
    stride_BC_pos,
    stride_state_indices_batch,
    stride_state_indices_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K_CACHE: tl.constexpr,
    BLOCK_SIZE_K_DOT: tl.constexpr,
    # heuristic-computed
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if HAS_STATE_BATCH_INDICES:
        state_batch_idx = tl.load(
            state_batch_indices_ptr
            + pid_b * stride_state_indices_batch
            + 0 * stride_state_indices_T
        ).to(tl.int64)
        if state_batch_idx == null_block_id:
            return
    else:
        state_batch_idx = pid_b

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    write_pos = tl.load(write_pos_ptr + pid_b).to(tl.int64)
    is_flush = tl.load(is_flush_ptr + pid_b) != 0

    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    x_cache_ptr += state_batch_idx * stride_xc_batch + pid_h * stride_xc_head
    dt_cache_ptr += state_batch_idx * stride_dtc_batch + pid_h * stride_dtc_head
    B_cache_ptr += (
        state_batch_idx * stride_Bc_batch
        + (pid_h // nheads_ngroups_ratio) * stride_Bc_group
    )
    BC_pre_ptr += (
        pid_b * stride_BC_batch
        + (pid_h // nheads_ngroups_ratio) * stride_BC_group
    )

    dt_cur = tl.load(dt_ptr).to(tl.float32)
    if HAS_DT_BIAS:
        dt_cur += tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
    if DT_SOFTPLUS:
        dt_cur = tl.where(dt_cur <= 20.0, softplus(dt_cur), dt_cur)

    A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
    x_cur = tl.load(x_ptr + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0)
    C = tl.load(
        C_ptr + offs_n * stride_C_dstate,
        mask=offs_n < dstate,
        other=0.0,
    ).to(tl.float32)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    state = tl.load(
        state_ptrs,
        mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
        other=0.0,
    )
    B_cur = tl.load(B_ptr + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0)

    if not is_flush:
        offs_k_cache = tl.arange(0, BLOCK_SIZE_K_CACHE)
        dt_all_cache = tl.load(
            dt_cache_ptr + offs_k_cache * stride_dtc_pos,
            mask=offs_k_cache < write_pos,
            other=0.0,
        ).to(tl.float32)
        dt_all_cache = tl.where(offs_k_cache == write_pos, dt_cur, dt_all_cache)

        dA_cumsum_cache = A * tl.cumsum(dt_all_cache, axis=0)
        dA_total_cache = A * tl.sum(dt_all_cache, axis=0)
        total_decay_cache = tl.exp(dA_total_cache)
        scale_cache = dt_all_cache * tl.exp(dA_total_cache - dA_cumsum_cache)
        scale_cache = tl.where(offs_k_cache <= write_pos, scale_cache, 0.0)

        x_all_cache_ptrs = (
            x_cache_ptr
            + offs_m[:, None] * stride_xc_dim
            + offs_k_cache[None, :] * stride_xc_pos
        )
        x_all_cache = tl.load(
            x_all_cache_ptrs,
            mask=(offs_m[:, None] < dim) & (offs_k_cache[None, :] < write_pos),
            other=0.0,
        )
        x_all_cache = tl.where(
            offs_k_cache[None, :] == write_pos, x_cur[:, None], x_all_cache
        )

        checkpoint_out = (
            tl.sum(state.to(tl.float32) * C[None, :], axis=1) * total_decay_cache
        )
        BC_cache = tl.load(
            BC_pre_ptr + offs_k_cache * stride_BC_pos,
            mask=offs_k_cache <= write_pos,
            other=0.0,
        )
        cache_out = tl.sum(
            x_all_cache.to(tl.float32) * (scale_cache * BC_cache)[None, :], axis=1
        )
        out = checkpoint_out + cache_out
        tl.store(
            x_cache_ptr + offs_m * stride_xc_dim + write_pos * stride_xc_pos,
            x_cur,
            mask=offs_m < dim,
        )
        if pid_m == 0:
            tl.store(dt_cache_ptr + write_pos * stride_dtc_pos, dt_cur)
            tl.store(
                B_cache_ptr + write_pos * stride_Bc_pos
                + offs_n * stride_Bc_dstate,
                B_cur,
                mask=offs_n < dstate,
            )
    else:
        offs_k_dot = tl.arange(0, BLOCK_SIZE_K_DOT)
        dt_all_dot = tl.load(
            dt_cache_ptr + offs_k_dot * stride_dtc_pos,
            mask=offs_k_dot < write_pos,
            other=0.0,
        ).to(tl.float32)
        dt_all_dot = tl.where(offs_k_dot == write_pos, dt_cur, dt_all_dot)

        dA_cumsum_dot = A * tl.cumsum(dt_all_dot, axis=0)
        dA_total_dot = A * tl.sum(dt_all_dot, axis=0)
        total_decay_dot = tl.exp(dA_total_dot)
        scale_dot = dt_all_dot * tl.exp(dA_total_dot - dA_cumsum_dot)
        scale_dot = tl.where(offs_k_dot <= write_pos, scale_dot, 0.0)

        x_all_dot_ptrs = (
            x_cache_ptr
            + offs_m[:, None] * stride_xc_dim
            + offs_k_dot[None, :] * stride_xc_pos
        )
        x_all_dot = tl.load(
            x_all_dot_ptrs,
            mask=(offs_m[:, None] < dim) & (offs_k_dot[None, :] < write_pos),
            other=0.0,
        )
        x_all_dot = tl.where(
            offs_k_dot[None, :] == write_pos, x_cur[:, None], x_all_dot
        )

        B_all_dot_ptrs = (
            B_cache_ptr
            + offs_k_dot[:, None] * stride_Bc_pos
            + offs_n[None, :] * stride_Bc_dstate
        )
        B_all_dot = tl.load(
            B_all_dot_ptrs,
            mask=(offs_k_dot[:, None] < write_pos) & (offs_n[None, :] < dstate),
            other=0.0,
        )
        B_all_dot = tl.where(
            offs_k_dot[:, None] == write_pos, B_cur[None, :], B_all_dot
        )

        B_scaled = (B_all_dot.to(tl.float32) * scale_dot[:, None]).to(
            x_ptr.dtype.element_ty
        )
        delta_state = tl.dot(x_all_dot.to(x_ptr.dtype.element_ty), B_scaled)
        state_new = state.to(tl.float32) * total_decay_dot + delta_state.to(tl.float32)
        tl.store(
            state_ptrs,
            state_new.to(state.dtype),
            mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
        )
        out = tl.sum(state_new * C[None, :], axis=1)

    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D = tl.load(
            D_ptr + offs_m * stride_D_dim,
            mask=offs_m < dim,
            other=0.0,
        ).to(tl.float32)
        out += x_cur.to(tl.float32) * D

    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
        z = tl.load(
            z_ptr + offs_m * stride_z_dim,
            mask=offs_m < dim,
            other=0.0,
        ).to(tl.float32)
        out *= z * tl.sigmoid(z)

    tl.store(out_ptr + offs_m * stride_out_dim, out, mask=offs_m < dim)


def _get_cached_bc_launch_config(dstate: int) -> tuple[int, int]:
    if dstate <= 64:
        return 32, 4
    if dstate <= 128:
        return 32, 2
    return 16, 8


def selective_state_update_cached_bc(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    x_cache: torch.Tensor | None = None,
    dt_cache: torch.Tensor | None = None,
    B_cache: torch.Tensor | None = None,
    bc_pre: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    is_flush: torch.Tensor | None = None,
    max_cache_len: int = 16,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cached-bc SSM update for vLLM's autoregressive Mamba2 decode path."""
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out is not None and out.dim() == 2:
        out = out.unsqueeze(1)
    if state_batch_indices is not None and state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    assert out is not None and out.shape == x.shape

    assert A.stride(-1) == 0 and A.stride(-2) == 0, (
        "Cached kernel requires TIE_HDIM (A scalar per head)"
    )
    assert dt.stride(-1) == 0, "Cached kernel requires TIE_HDIM (dt scalar per head)"
    if dt_bias is not None:
        assert dt_bias.stride(-1) == 0, (
            "Cached kernel requires TIE_HDIM (dt_bias scalar per head)"
        )

    assert x_cache is not None
    assert dt_cache is not None
    assert B_cache is not None
    assert x_cache.shape[1:] == (nheads, max_cache_len, dim)
    assert dt_cache.shape[1:] == (nheads, max_cache_len)
    assert B_cache.shape[1:] == (ngroups, max_cache_len, dstate)
    assert write_pos is not None and write_pos.shape[0] >= batch
    assert write_pos.dtype == torch.int32
    assert is_flush is not None and is_flush.shape[0] >= batch
    assert is_flush.dtype in (torch.bool, torch.int8)
    assert bc_pre is not None
    assert bc_pre.shape[0] >= batch and bc_pre.shape[1] >= ngroups
    assert bc_pre.shape[2] == max_cache_len
    assert bc_pre.dtype == torch.float32
    if state_batch_indices is not None:
        assert state_batch_indices.shape[0] >= batch
        assert state_batch_indices.shape[1] >= 1

    block_size_k_cache = max(1, triton.next_power_of_2(max_cache_len))
    block_size_k_dot = max(16, block_size_k_cache)
    block_size_m, num_warps = _get_cached_bc_launch_config(dstate)

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    state_indices_strides = (
        (state_batch_indices.stride(0), state_batch_indices.stride(1))
        if state_batch_indices is not None
        else (0, 0)
    )

    with torch.accelerator.device_index(x.device.index):
        _cached_bc_precompute_kernel[(batch, ngroups)](
            B,
            C,
            B_cache,
            write_pos,
            is_flush,
            bc_pre,
            state_batch_indices,
            null_block_id,
            batch,
            ngroups,
            dstate,
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            B_cache.stride(0),
            B_cache.stride(1),
            B_cache.stride(2),
            B_cache.stride(3),
            bc_pre.stride(0),
            bc_pre.stride(1),
            bc_pre.stride(2),
            state_indices_strides[0],
            state_indices_strides[1],
            max_cache_len,
            block_size_k_cache,
            num_warps=2,
        )
        _selective_scan_update_cached_bc_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            x_cache,
            dt_cache,
            B_cache,
            bc_pre,
            write_pos,
            is_flush,
            state_batch_indices,
            null_block_id,
            batch,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            x_cache.stride(0),
            x_cache.stride(1),
            x_cache.stride(3),
            x_cache.stride(2),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            B_cache.stride(0),
            B_cache.stride(1),
            B_cache.stride(2),
            B_cache.stride(3),
            bc_pre.stride(0),
            bc_pre.stride(1),
            bc_pre.stride(2),
            state_indices_strides[0],
            state_indices_strides[1],
            dt_softplus,
            max_cache_len,
            block_size_m,
            block_size_k_cache,
            block_size_k_dot,
            num_warps=num_warps,
        )

    if not has_heads:
        out = out.squeeze(1)
    return out


def selective_state_update_cached_bc_ref(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    x_cache: torch.Tensor | None = None,
    dt_cache: torch.Tensor | None = None,
    B_cache: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    max_cache_len: int = 16,
) -> torch.Tensor:
    """Pure-PyTorch cached-bc reference for validation."""
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape

    ratio = nheads // ngroups

    dt_val = dt[:, :, 0].float()
    if dt_bias is not None:
        dt_val = dt_val + dt_bias[:, 0].float()
    if dt_softplus:
        dt_val = F.softplus(dt_val)
    A_val = A[:, 0, 0].float()
    C_heads = C.repeat_interleave(ratio, dim=1)
    out = torch.empty(batch, nheads, dim, device=x.device, dtype=torch.float32)

    for b in range(batch):
        cache_len = int(write_pos[b].item())
        is_flush = cache_len == max_cache_len - 1
        n_steps = cache_len + 1

        dt_all = torch.zeros(nheads, n_steps, device=x.device, dtype=torch.float32)
        if cache_len > 0:
            dt_all[:, :cache_len] = dt_cache[b, :, :cache_len]
        dt_all[:, cache_len] = dt_val[b]

        cumsum = torch.cumsum(dt_all, dim=-1)
        total = cumsum[:, -1]
        dA_cumsum = A_val[:, None] * cumsum
        dA_total = A_val * total
        total_decay = torch.exp(dA_total)
        scale = dt_all * torch.exp(dA_total[:, None] - dA_cumsum)

        x_all = torch.zeros(nheads, dim, n_steps, device=x.device, dtype=x.dtype)
        if cache_len > 0:
            x_all[..., :cache_len] = x_cache[b, :, :cache_len, :].permute(0, 2, 1)
        x_all[..., cache_len] = x[b]

        B_all = torch.zeros(ngroups, n_steps, dstate, device=B.device, dtype=B.dtype)
        if cache_len > 0:
            B_all[:, :cache_len, :] = B_cache[b, :, :cache_len, :]
        B_all[:, cache_len, :] = B[b]

        B_heads = B_all.repeat_interleave(ratio, dim=0)
        C_heads_b = C_heads[b]

        if is_flush:
            x_scaled = x_all.float() * scale[:, None, :]
            delta = torch.einsum("hdk,hkn->hdn", x_scaled, B_heads.float())
            state_new = state[b].float() * total_decay[:, None, None] + delta
            state[b].copy_(state_new.to(state.dtype))
            out[b] = torch.einsum("hdn,hn->hd", state_new, C_heads_b.float())
        else:
            checkpoint_out = torch.einsum(
                "hdn,hn->hd", state[b].float(), C_heads_b.float()
            )
            checkpoint_out = checkpoint_out * total_decay[:, None]
            BC = torch.einsum("hkn,hn->hk", B_heads.float(), C_heads_b.float())
            cache_out = torch.einsum("hdk,hk->hd", x_all.float(), scale * BC)
            out[b] = checkpoint_out + cache_out
            x_cache[b, :, cache_len, :] = x[b]
            dt_cache[b, :, cache_len] = dt_val[b]
            B_cache[b, :, cache_len, :] = B[b]

    if D is not None:
        out = out + (x.float() * D[None]).to(out.dtype)
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out
