# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


# ======================================================================
# Fused scatter + precompute  (grid: (batch, ngroups))
#
# Scatters all conv_dim channels (x|B|C, partitioned by group) + dt of the
# fresh spec tokens into the circular post-conv / dt caches at
# ``(origin + write_pos + s) % buf``, and computes ``bc[k, s] = B_full[k] . C[s]``
# over the window (history B from the cache + fresh spec B, no read-back).
# ======================================================================
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _fused_scatter_precompute_kernel(
    conv_out_ptr,  # (total_tokens, conv_dim) packed channel-last conv output
    dt_spec_ptr,  # (total_tokens, nheads) packed raw dt
    post_conv_cache_ptr,  # (num_blocks, buf, conv_dim) circular paged
    dt_cache_ptr,  # (num_blocks, nheads, buf) circular paged
    write_pos_ptr,  # (num_state_slots,) block-keyed
    post_origin_ptr,  # (num_state_slots,) block-keyed
    bc_pre_ptr,  # (max_bs, ngroups, max_cache_len, block_spec) dense per-row scratch
    state_batch_indices_ptr,  # (batch,) physical block per dense decode row
    query_start_loc_ptr,  # (batch + 1,) packed token offsets
    null_block_id,
    batch,
    ngroups,
    nheads,
    dstate,
    d_inner,
    conv_dim,
    max_cache_len,
    stride_conv_out_tok,
    stride_conv_out_c,
    stride_dt_spec_tok,
    stride_dt_spec_h,
    stride_post_conv_cache_b,
    stride_post_conv_cache_pos,
    stride_post_conv_cache_c,
    stride_dt_cache_b,
    stride_dt_cache_h,
    stride_dt_cache_pos,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    stride_bc_pre_pos,
    stride_bc_pre_spec,
    stride_state_indices_batch,
    RATIO: tl.constexpr,
    RATIO_P: tl.constexpr,
    NCX: tl.constexpr,
    BLOCK_CX: tl.constexpr,
    CACHE_BUF_LEN: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    BLOCK_SIZE_SPEC: tl.constexpr,
    BLOCK_HL: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    state_batch_idx = tl.load(
        state_batch_indices_ptr + pid_b * stride_state_indices_batch
    ).to(tl.int64)
    if state_batch_idx == null_block_id:
        return
    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    spec_len = (eos - bos).to(tl.int32)
    write_pos = tl.load(write_pos_ptr + state_batch_idx).to(tl.int32)
    post_origin = tl.load(post_origin_ptr + state_batch_idx).to(tl.int32)

    offs_s = tl.arange(0, BLOCK_SIZE_SPEC)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    spec_valid = offs_s < spec_len
    nmask = offs_n < dstate
    phys_spec = (post_origin + write_pos + offs_s) & (CACHE_BUF_LEN - 1)

    b_c0 = d_inner + pid_g * dstate
    c_c0 = d_inner + ngroups * dstate + pid_g * dstate

    src_base = conv_out_ptr + bos * stride_conv_out_tok
    # fresh spec B / C  [S, N]
    B_spec = tl.load(
        src_base + (b_c0 + offs_n[None, :]) * stride_conv_out_c + offs_s[:, None] * stride_conv_out_tok,
        mask=spec_valid[:, None] & nmask[None, :],
        other=0.0,
    )
    C_spec = tl.load(
        src_base + (c_c0 + offs_n[None, :]) * stride_conv_out_c + offs_s[:, None] * stride_conv_out_tok,
        mask=spec_valid[:, None] & nmask[None, :],
        other=0.0,
    )
    cache_base = post_conv_cache_ptr + state_batch_idx * stride_post_conv_cache_b
    # scatter B / C into circular cache
    tl.store(
        cache_base + phys_spec[:, None] * stride_post_conv_cache_pos + (b_c0 + offs_n[None, :]) * stride_post_conv_cache_c,
        B_spec,
        mask=spec_valid[:, None] & nmask[None, :],
    )
    tl.store(
        cache_base + phys_spec[:, None] * stride_post_conv_cache_pos + (c_c0 + offs_n[None, :]) * stride_post_conv_cache_c,
        C_spec,
        mask=spec_valid[:, None] & nmask[None, :],
    )

    # scatter x channels owned by this group: [g*RATIO_P, (g+1)*RATIO_P)
    gx0 = pid_g * RATIO_P
    for i in tl.static_range(NCX):
        offs_cx = i * BLOCK_CX + tl.arange(0, BLOCK_CX)
        cxm = offs_cx < RATIO_P
        gx = gx0 + offs_cx
        xv = tl.load(
            src_base
            + gx[None, :] * stride_conv_out_c
            + offs_s[:, None] * stride_conv_out_tok,
            mask=spec_valid[:, None] & cxm[None, :],
            other=0.0,
        )
        tl.store(
            cache_base
            + phys_spec[:, None] * stride_post_conv_cache_pos
            + gx[None, :] * stride_post_conv_cache_c,
            xv,
            mask=spec_valid[:, None] & cxm[None, :],
        )

    # scatter dt for this group's heads
    offs_hl = tl.arange(0, BLOCK_HL)
    hlm = offs_hl < RATIO
    gh = pid_g * RATIO + offs_hl
    dt_base = dt_spec_ptr + bos * stride_dt_spec_tok
    dtv = tl.load(
        dt_base + offs_s[:, None] * stride_dt_spec_tok + gh[None, :] * stride_dt_spec_h,
        mask=spec_valid[:, None] & hlm[None, :],
        other=0.0,
    )
    dtc_base = dt_cache_ptr + state_batch_idx * stride_dt_cache_b
    tl.store(
        dtc_base
        + gh[None, :] * stride_dt_cache_h
        + phys_spec[:, None] * stride_dt_cache_pos,
        dtv,
        mask=spec_valid[:, None] & hlm[None, :],
    )

    # bc: history B from cache + fresh spec B  (no read-back of spec B)
    offs_k = tl.arange(0, BLOCK_SIZE_CACHE)
    hist_mask = offs_k < write_pos
    cache_valid = (offs_k < max_cache_len) & (offs_k < (write_pos + spec_len))
    spec_tok = (offs_k >= write_pos) & (offs_k < (write_pos + spec_len))
    spec_off = offs_k - write_pos
    phys_k = (post_origin + offs_k) & (CACHE_BUF_LEN - 1)
    B_hist = tl.load(
        cache_base + phys_k[:, None] * stride_post_conv_cache_pos + (b_c0 + offs_n[None, :]) * stride_post_conv_cache_c,
        mask=hist_mask[:, None] & nmask[None, :],
        other=0.0,
    )
    B_specrows = tl.load(
        src_base + (b_c0 + offs_n[None, :]) * stride_conv_out_c + spec_off[:, None] * stride_conv_out_tok,
        mask=spec_tok[:, None] & nmask[None, :],
        other=0.0,
    )
    B_full = tl.where(spec_tok[:, None], B_specrows, B_hist)
    B_full = tl.where(cache_valid[:, None], B_full.to(tl.float32), 0.0).to(
        conv_out_ptr.dtype.element_ty
    )
    bc = tl.dot(
        B_full,
        tl.trans(C_spec.to(conv_out_ptr.dtype.element_ty)),
        input_precision="tf32x3",
    ).to(tl.float32)
    bc_ptrs = (
        bc_pre_ptr
        + pid_b * stride_bc_pre_batch
        + pid_g * stride_bc_pre_group
        + offs_k[:, None] * stride_bc_pre_pos
        + offs_s[None, :] * stride_bc_pre_spec
    )
    tl.store(
        bc_ptrs,
        bc.to(bc_pre_ptr.dtype.element_ty),
        mask=cache_valid[:, None] & spec_valid[None, :],
    )


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _replayssm_spec_circular_kernel(
    state_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    C_full_ptr,
    bc_pre_ptr,
    D_ptr,
    z_ptr,
    dt_bias_ptr,
    A_ptr,
    out_ptr,
    is_flush_flags_ptr,
    write_pos_ptr,
    post_origin_ptr,
    state_batch_indices_ptr,
    query_start_loc_ptr,
    null_block_id,
    batch,
    nheads,
    dim,
    dstate,
    max_cache_len,
    nheads_ngroups_ratio,
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_cache_batch,
    stride_x_cache_head,
    stride_x_cache_dim,
    stride_x_cache_pos,
    stride_dt_cache_batch,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_batch,
    stride_B_cache_group,
    stride_B_cache_dstate,
    stride_B_cache_pos,
    stride_C_full_batch,
    stride_C_full_group,
    stride_C_full_dstate,
    stride_C_full_pos,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    stride_bc_pre_pos,
    stride_bc_pre_spec,
    stride_D_head,
    stride_D_dim,
    stride_z_tok,
    stride_z_head,
    stride_z_dim,
    stride_dt_bias_head,
    stride_A_head,
    stride_out_tok,
    stride_out_head,
    stride_out_dim,
    stride_state_indices_batch,
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    BLOCK_SIZE_SPEC: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    CACHE_BUF_LEN: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Resolve the physical state slot for this decode row; skip padded rows.
    state_batch_idx = tl.load(state_batch_indices_ptr + pid_b * stride_state_indices_batch).to(tl.int64)
    if state_batch_idx == null_block_id:
        return

    # This row's draft tokens live in [bos, eos); spec_len is the spec window.
    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    spec_len = (eos - bos).to(tl.int32)

    # Block-keyed cursors: flush flag, buffer cursor, and ring-buffer origin.
    is_flush = tl.load(is_flush_flags_ptr + state_batch_idx) != 0
    write_pos = tl.load(write_pos_ptr + state_batch_idx).to(tl.int32)
    post_origin = tl.load(post_origin_ptr + state_batch_idx).to(tl.int32)

    # Axes: offs_m=headdim, offs_n=dstate, offs_k=cache window, offs_s=draft pos.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_k = tl.arange(0, BLOCK_SIZE_CACHE)
    offs_s = tl.arange(0, BLOCK_SIZE_SPEC)
    spec_valid_mask = offs_s < spec_len
    hist_mask = offs_k < write_pos
    cache_valid_mask = (offs_k < max_cache_len) & (offs_k < (write_pos + spec_len))
    spec_token_mask = (offs_k >= write_pos) & (offs_k < (write_pos + spec_len))
    spec_cache_pos = write_pos + offs_s
    spec_prefix_mask = spec_valid_mask[:, None] & spec_token_mask[None, :] & (offs_k[None, :] <= spec_cache_pos[:, None])
    # Ring-buffer physical positions for the window and for the draft tokens.
    phys_k = (post_origin + offs_k) & (CACHE_BUF_LEN - 1)
    phys_spec = (post_origin + spec_cache_pos) & (CACHE_BUF_LEN - 1)

    # Advance pointers to this (row, head, group).
    state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    x_cache_ptr += state_batch_idx * stride_x_cache_batch + pid_h * stride_x_cache_head
    dt_cache_ptr += state_batch_idx * stride_dt_cache_batch + pid_h * stride_dt_cache_head
    B_cache_ptr += state_batch_idx * stride_B_cache_batch + (pid_h // nheads_ngroups_ratio) * stride_B_cache_group
    C_full_ptr += state_batch_idx * stride_C_full_batch + (pid_h // nheads_ngroups_ratio) * stride_C_full_group
    bc_pre_ptr += pid_b * stride_bc_pre_batch + (pid_h // nheads_ngroups_ratio) * stride_bc_pre_group
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    if HAS_Z:
        z_ptr += bos * stride_z_tok + pid_h * stride_z_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    out_ptr += bos * stride_out_tok + pid_h * stride_out_head

    # dt over the window from the (ring) cache, with bias / softplus, masked to valid.
    A_val = tl.load(A_ptr).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr).to(tl.float32) if HAS_DT_BIAS else 0.0
    dt_cache_block = tl.load(dt_cache_ptr + phys_k * stride_dt_cache_pos, mask=cache_valid_mask, other=0.0).to(tl.float32)
    dt_cache_block = tl.where(cache_valid_mask, dt_cache_block, 0.0)
    if HAS_DT_BIAS:
        dt_cache_block = tl.where(cache_valid_mask, dt_cache_block + dt_bias_val, 0.0)
    if DT_SOFTPLUS:
        dt_cache_block = tl.where(cache_valid_mask, tl.where(dt_cache_block <= 20.0, softplus(dt_cache_block), dt_cache_block), 0.0)

    # Decay weights per draft: cumulative dt, committed-history total, per-draft
    # prefix sum, and the checkpoint decay applied to the S_0 readout.
    dt_cum = tl.cumsum(dt_cache_block, axis=0)
    hist_total = tl.sum(tl.where(hist_mask, dt_cache_block, 0.0), axis=0)
    spec_cum = tl.sum(tl.where(spec_prefix_mask, dt_cache_block[None, :], 0.0), axis=1)
    spec_cum = tl.where(spec_valid_mask, spec_cum, 0.0)
    spec_total = hist_total + spec_cum
    checkpoint_decay = tl.where(spec_valid_mask, tl.exp(tl.minimum(A_val * spec_total, 0.0)), 0.0)

    # Checkpoint state S_0, cached values x, precomputed k^T q (bc), per-draft q.
    state_ptrs = state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    state_f = state.to(tl.float32)
    x_cache_block = tl.load(x_cache_ptr + phys_k[None, :] * stride_x_cache_pos + offs_m[:, None] * stride_x_cache_dim, mask=(offs_m[:, None] < dim) & cache_valid_mask[None, :], other=0.0)
    x_cache_ty = x_cache_block.to(x_cache_ptr.dtype.element_ty)
    state_ty = state_f.to(x_cache_ptr.dtype.element_ty)
    bc = tl.load(bc_pre_ptr + offs_k[:, None] * stride_bc_pre_pos + offs_s[None, :] * stride_bc_pre_spec, mask=cache_valid_mask[:, None] & spec_valid_mask[None, :], other=0.0).to(tl.float32)
    C_load_mask = spec_valid_mask[:, None] & (offs_n[None, :] < dstate) & (spec_cache_pos[:, None] < max_cache_len)
    C_spec = tl.load(C_full_ptr + phys_spec[:, None] * stride_C_full_pos + offs_n[None, :] * stride_C_full_dstate, mask=C_load_mask, other=0.0).to(tl.float32)
    C_spec_ty = C_spec.to(x_cache_ptr.dtype.element_ty)

    # Per-draft output-only readout: decayed checkpoint term (S_0 q_s) ...
    checkpoint_out = tl.dot(state_ty, tl.trans(C_spec_ty), input_precision="tf32x3").to(tl.float32)
    checkpoint_out *= checkpoint_decay[None, :]
    # ... plus the causal weighted sum over cached values via the k^T q GEMM.
    spec_scale_mat = dt_cache_block[:, None] * tl.exp(tl.minimum(A_val * (spec_total[None, :] - dt_cum[:, None]), 0.0))
    causal = spec_valid_mask[None, :] & cache_valid_mask[:, None] & (offs_k[:, None] <= spec_cache_pos[None, :])
    factor = tl.where(causal, bc * spec_scale_mat, 0.0)
    spec_contrib = tl.dot(x_cache_ty, factor.to(x_cache_ptr.dtype.element_ty), input_precision="tf32x3").to(tl.float32)
    out = tl.trans(checkpoint_out + spec_contrib)

    # Skip connection (D) and output gate (z), per draft token.
    if HAS_D:
        x_spec = tl.load(x_cache_ptr + offs_m[None, :] * stride_x_cache_dim + phys_spec[:, None] * stride_x_cache_pos, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim), other=0.0).to(tl.float32)
        D_val = tl.load(D_ptr + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        out += x_spec * D_val[None, :]
    if HAS_Z:
        z_val = tl.load(z_ptr + offs_s[:, None] * stride_z_tok + offs_m[None, :] * stride_z_dim, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim), other=0.0).to(tl.float32)
        out *= z_val * tl.sigmoid(z_val)

    # Write the per-draft outputs into the packed out buffer.
    out = tl.where(spec_valid_mask[:, None], out, 0.0)
    tl.store(out_ptr + offs_s[:, None] * stride_out_tok + offs_m[None, :] * stride_out_dim, out, mask=spec_valid_mask[:, None] & (offs_m[None, :] < dim))

    # Flush step: fold ONLY the committed history (not the speculative tokens,
    # which may still be rolled back) into the checkpoint state and store it.
    if is_flush:
        B_block = tl.load(B_cache_ptr + phys_k[:, None] * stride_B_cache_pos + offs_n[None, :] * stride_B_cache_dstate, mask=cache_valid_mask[:, None] & (offs_n[None, :] < dstate), other=0.0)
        B_f = tl.where(cache_valid_mask[:, None], B_block.to(tl.float32), 0.0)
        hist_decay = tl.exp(tl.minimum(A_val * hist_total, 0.0))
        hist_scale = tl.where(hist_mask, dt_cache_block * tl.exp(tl.minimum(A_val * (hist_total - dt_cum), 0.0)), 0.0)
        B_hist_scaled = (B_f * hist_scale[:, None]).to(x_cache_ptr.dtype.element_ty)
        delta_state = tl.dot(x_cache_ty, B_hist_scaled, input_precision="tf32x3")
        replay_state = state_f * hist_decay + delta_state.to(tl.float32)
        if write_pos > 0:
            tl.store(state_ptrs, replay_state.to(state.dtype), mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))


@triton.jit
def _advance_write_pos_origin_kernel(
    write_pos_ptr,
    post_origin_ptr,
    is_flush_ptr,
    num_accepted_ptr,
    state_batch_indices_ptr,
    null_block_id,
    batch,
    stride_state_indices_batch,
    MAX_CACHE_LEN: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
    CACHE_BUF_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    row_mask = offs < batch
    state_batch_idx = tl.load(state_batch_indices_ptr + offs * stride_state_indices_batch, mask=row_mask, other=null_block_id).to(
        tl.int64
    )
    valid = row_mask & (state_batch_idx != null_block_id)
    write_pos = tl.load(
        write_pos_ptr + state_batch_idx, mask=valid, other=0
    ).to(tl.int32)
    post_origin = tl.load(
        post_origin_ptr + state_batch_idx, mask=valid, other=0
    ).to(tl.int32)
    is_flush_cur = tl.load(
        is_flush_ptr + state_batch_idx, mask=valid, other=0
    ).to(tl.int32)
    num_accepted = tl.load(num_accepted_ptr + offs, mask=valid, other=0).to(tl.int32)
    total_commit = tl.where(valid, num_accepted, 0).to(tl.int32)
    flush_now = (total_commit > 0) & (is_flush_cur != 0)
    new_origin = tl.where(
        flush_now, (post_origin + write_pos) & (CACHE_BUF_LEN - 1), post_origin
    ).to(tl.int32)
    new_wp = tl.where(
        total_commit <= 0,
        write_pos,
        tl.where(is_flush_cur != 0, total_commit, write_pos + total_commit),
    ).to(tl.int32)
    # EARLY-FLUSH (margin = 2 * MAX_SPEC_LEN, strict '>'): flush one window early
    # so that on EVERY verify step write_pos + spec_len <= max_cache_len. The
    # strict '>' uses the buffer exactly -- the largest write_pos at a flush step
    # is max_cache_len - max_spec_len, whose window fills the last slots with zero
    # headroom. Config enforces max_cache_len >= 2 * max_spec_len.
    next_is_flush = ((new_wp + 2 * MAX_SPEC_LEN) > MAX_CACHE_LEN).to(tl.int8)
    tl.store(post_origin_ptr + state_batch_idx, new_origin, mask=valid)
    tl.store(write_pos_ptr + state_batch_idx, new_wp, mask=valid)
    tl.store(is_flush_ptr + state_batch_idx, next_is_flush, mask=valid)


@triton.jit
def _reset_replayssm_spec_cursors_kernel(
    write_pos_ptr,
    post_origin_ptr,
    is_flush_ptr,
    first_decode_ptr,  # (batch,) int8 mask
    state_batch_indices_ptr,
    null_block_id,
    batch,
    stride_state_indices_batch,
    INIT_IS_FLUSH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    row_mask = offs < batch
    state_batch_idx = tl.load(state_batch_indices_ptr + offs * stride_state_indices_batch, mask=row_mask, other=null_block_id).to(
        tl.int64
    )
    first = tl.load(first_decode_ptr + offs, mask=row_mask, other=0).to(tl.int32)
    do_reset = row_mask & (state_batch_idx != null_block_id) & (first != 0)
    tl.store(
        write_pos_ptr + state_batch_idx,
        tl.zeros_like(state_batch_idx).to(tl.int32),
        mask=do_reset,
    )
    tl.store(
        post_origin_ptr + state_batch_idx,
        tl.zeros_like(state_batch_idx).to(tl.int32),
        mask=do_reset,
    )
    tl.store(
        is_flush_ptr + state_batch_idx,
        (tl.zeros_like(state_batch_idx) + INIT_IS_FLUSH).to(tl.int8),
        mask=do_reset,
    )


def _get_spec_launch_config(dstate: int, max_spec_len: int) -> tuple[int, int]:
    """Config sweep is strongly recommended for different dstate, spec_len and hardware"""
    block_size_m = 16
    if dstate <= 64:
        num_warps = 2 if max_spec_len in (1, 8) else 1
    elif dstate <= 128:
        num_warps = 2 if max_spec_len in (1, 8) else 1
    else:
        num_warps = 2 if max_spec_len in (1, 8) else 1
    return block_size_m, num_warps


def selective_state_update_replayssm_spec(
    state_checkpoint: torch.Tensor,  # (num_blocks, H, P, N) checkpoint (flush updates in place)
    post_conv_cache: torch.Tensor,  # (num_blocks, cache_buf_len, conv_dim) circular
    dt_cache: torch.Tensor,  # (num_blocks, H, cache_buf_len) circular
    conv_out: torch.Tensor,  # (total_tokens, conv_dim) packed channel-last post-conv
    dt_spec: torch.Tensor,  # (total_tokens, H) packed raw dt
    A: torch.Tensor,  # (H, P, N) TIE_HDIM (A.stride(-1)==A.stride(-2)==0)
    write_pos: torch.Tensor,  # (num_state_slots,) int32 block-keyed cursor
    post_conv_state_pos: torch.Tensor,  # (num_state_slots,) int32 circular origin
    is_flush: torch.Tensor,  # (num_state_slots,) int8 block-keyed flag
    query_start_loc: torch.Tensor,  # (batch + 1,) int32 packed offsets
    state_batch_indices: torch.Tensor,  # (batch,) int32 physical block per row
    max_cache_len: int,
    max_spec_len: int,
    d_inner: int,
    ngroups: int,
    dstate: int,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    out: torch.Tensor | None = None,
    bc_pre: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    force_block_size_m: int | None = None,
    force_num_warps: int | None = None,
) -> torch.Tensor:
    """One Mamba2 speculative verify step on the paged CIRCULAR post-conv cache.

    Hybrid conv variant: ``conv_out`` is the post-conv output of vLLM's
    ``causal_conv1d_update`` (packed channel-last ``[total_tokens, conv_dim]``).
    Fuses scatter + bc-precompute in one ``(batch, ngroups)`` launch, then runs
    the circular scan. Cursors are block-keyed (indexed by
    ``state_batch_indices``); the commit (``commit_replayssm_spec``) advances them
    once per step. Writes into the caller-supplied packed ``out``
    ``[total_tokens, H, P]``.
    """
    num_blocks, nheads, dim, n_state = state_checkpoint.shape
    assert n_state == dstate
    total_tokens, conv_dim = conv_out.shape
    buf = post_conv_cache.shape[1]
    cache_buf_len = buf
    assert cache_buf_len & (cache_buf_len - 1) == 0, "cache_buf_len must be a power of two"
    assert d_inner == nheads * dim
    assert post_conv_cache.shape == (num_blocks, buf, conv_dim)
    assert dt_cache.shape == (num_blocks, nheads, buf)
    assert dt_spec.shape == (total_tokens, nheads)
    assert A.shape == (nheads, dim, dstate) and A.stride(-1) == 0 and A.stride(-2) == 0
    batch = state_batch_indices.shape[0]
    assert query_start_loc.shape[0] == batch + 1
    max_cache_len = buf

    if out is None:
        out = torch.empty(total_tokens, nheads, dim, device=conv_out.device, dtype=conv_out.dtype)
    if total_tokens == 0:
        return out

    block_spec = max(1, triton.next_power_of_2(max_spec_len))
    block_cache = max(16, triton.next_power_of_2(max_cache_len))
    block_size_m, num_warps = _get_spec_launch_config(dstate, max_spec_len)
    if force_block_size_m is not None:
        block_size_m = force_block_size_m
    if force_num_warps is not None:
        num_warps = force_num_warps

    if bc_pre is None:
        bc_pre = torch.empty(
            batch, ngroups, max_cache_len, block_spec, device=conv_out.device, dtype=conv_out.dtype
        )
    state_indices_stride = state_batch_indices.stride(0)

    # --- fused scatter + precompute ---
    ratio = nheads // ngroups
    ratio_p = ratio * dim
    BLOCK_CX = 256
    NCX = triton.cdiv(ratio_p, BLOCK_CX)
    block_hl = max(1, triton.next_power_of_2(ratio))
    with torch.cuda.device(conv_out.device.index):
        _fused_scatter_precompute_kernel[(batch, ngroups)](
            conv_out,
            dt_spec,
            post_conv_cache,
            dt_cache,
            write_pos,
            post_conv_state_pos,
            bc_pre,
            state_batch_indices,
            query_start_loc,
            null_block_id,
            batch,
            ngroups,
            nheads,
            dstate,
            d_inner,
            conv_dim,
            max_cache_len,
            conv_out.stride(0),
            conv_out.stride(1),
            dt_spec.stride(0),
            dt_spec.stride(1),
            post_conv_cache.stride(0),
            post_conv_cache.stride(1),
            post_conv_cache.stride(2),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            bc_pre.stride(0),
            bc_pre.stride(1),
            bc_pre.stride(2),
            bc_pre.stride(3),
            state_indices_stride,
            RATIO=ratio,
            RATIO_P=ratio_p,
            NCX=NCX,
            BLOCK_CX=BLOCK_CX,
            CACHE_BUF_LEN=cache_buf_len,
            BLOCK_SIZE_CACHE=block_cache,
            BLOCK_SIZE_SPEC=block_spec,
            BLOCK_HL=block_hl,
            num_warps=4,
        )

    # views into the paged circular post-conv cache: x | B | C on the channel axis
    x_view = (
        post_conv_cache[:, :, :d_inner]
        .view(num_blocks, buf, nheads, dim)
        .permute(0, 2, 1, 3)
    )
    B_view = (
        post_conv_cache[:, :, d_inner : d_inner + ngroups * dstate]
        .view(num_blocks, buf, ngroups, dstate)
        .permute(0, 2, 1, 3)
    )
    C_view = (
        post_conv_cache[:, :, d_inner + ngroups * dstate :]
        .view(num_blocks, buf, ngroups, dstate)
        .permute(0, 2, 1, 3)
    )

    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    )
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    with torch.cuda.device(state_checkpoint.device.index):
        _replayssm_spec_circular_kernel[grid](
            state_checkpoint,
            x_view,
            dt_cache,
            B_view,
            C_view,
            bc_pre,
            D,
            z,
            dt_bias,
            A,
            out,
            is_flush,
            write_pos,
            post_conv_state_pos,
            state_batch_indices,
            query_start_loc,
            null_block_id,
            batch,
            nheads,
            dim,
            dstate,
            max_cache_len,
            ratio,
            state_checkpoint.stride(0),
            state_checkpoint.stride(1),
            state_checkpoint.stride(2),
            state_checkpoint.stride(3),
            x_view.stride(0),
            x_view.stride(1),
            x_view.stride(3),
            x_view.stride(2),
            dt_cache.stride(0),
            dt_cache.stride(1),
            dt_cache.stride(2),
            B_view.stride(0),
            B_view.stride(1),
            B_view.stride(3),
            B_view.stride(2),
            C_view.stride(0),
            C_view.stride(1),
            C_view.stride(3),
            C_view.stride(2),
            bc_pre.stride(0),
            bc_pre.stride(1),
            bc_pre.stride(2),
            bc_pre.stride(3),
            D.stride(0) if D is not None else 0,
            D.stride(1) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            state_indices_stride,
            dt_softplus,
            block_size_m,
            block_cache,
            block_spec,
            CACHE_BUF_LEN=cache_buf_len,
            num_warps=num_warps,
        )
    return out


def commit_replayssm_spec(
    write_pos: torch.Tensor,
    post_conv_state_pos: torch.Tensor,
    is_flush: torch.Tensor,
    num_accepted_tokens: torch.Tensor,  # (batch,) int32, INCLUDES bonus (min 1)
    state_batch_indices: torch.Tensor,  # (batch,) int32
    max_cache_len: int,
    max_spec_len: int,
    cache_buf_len: int | None = None,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """CUDA-graph-safe block-keyed commit. Advances ``write_pos`` and the
    circular origin ``post_conv_state_pos`` (flush = O(1) bump, no relocate) for
    each decode row's physical block, and precomputes next-step ``is_flush``.
    Maps vLLM ``num_accepted_tokens`` (incl. bonus) to ``total_commit`` directly.
    Hybrid: no ``conv_seq_pos`` (the conv commit lives in causal_conv1d_update)."""
    batch = state_batch_indices.shape[0]
    if cache_buf_len is None:
        cache_buf_len = max(1, triton.next_power_of_2(max_cache_len))
    BLOCK = max(1, triton.next_power_of_2(batch))
    with torch.cuda.device(write_pos.device.index):
        _advance_write_pos_origin_kernel[(1,)](
            write_pos,
            post_conv_state_pos,
            is_flush,
            num_accepted_tokens,
            state_batch_indices,
            null_block_id,
            batch,
            state_batch_indices.stride(0),
            MAX_CACHE_LEN=max_cache_len,
            MAX_SPEC_LEN=max_spec_len,
            CACHE_BUF_LEN=cache_buf_len,
            BLOCK_SIZE=BLOCK,
            num_warps=1,
        )


def reset_replayssm_spec_cursors(
    write_pos: torch.Tensor,
    post_conv_state_pos: torch.Tensor,
    is_flush: torch.Tensor,
    first_decode_mask: torch.Tensor,  # (batch,) int8
    state_batch_indices: torch.Tensor,  # (batch,) int32
    max_cache_len: int,
    max_spec_len: int,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Prefill->decode reset for first-decode rows (block-keyed, per-request).
    Hybrid: cursors only -- no pre_conv_cache seed (conv_state carries context)."""
    batch = state_batch_indices.shape[0]
    BLOCK = max(1, triton.next_power_of_2(batch))
    # Early-flush margin (2 * max_spec_len, strict '>'): match
    # _advance_write_pos_origin_kernel so the first decode after prefill agrees
    # with the steady-state flush cadence.
    init_is_flush = 1 if 2 * max_spec_len > max_cache_len else 0
    with torch.cuda.device(write_pos.device.index):
        _reset_replayssm_spec_cursors_kernel[(1,)](
            write_pos,
            post_conv_state_pos,
            is_flush,
            first_decode_mask,
            state_batch_indices,
            null_block_id,
            batch,
            state_batch_indices.stride(0),
            INIT_IS_FLUSH=init_is_flush,
            BLOCK_SIZE=BLOCK,
            num_warps=1,
        )


__all__ = [
    "selective_state_update_replayssm_spec",
    "commit_replayssm_spec",
    "reset_replayssm_spec_cursors",
]
