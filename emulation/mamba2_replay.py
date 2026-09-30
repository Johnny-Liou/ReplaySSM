# SPDX-License-Identifier: Apache-2.0
"""Mamba2 output replay and independent checkpoint/read-copy quantization."""

from __future__ import annotations

import triton
import triton.language as tl

from . import quant_tl

PREC_FP32 = tl.constexpr(0)
PREC_FP16 = tl.constexpr(1)
PREC_FP8 = tl.constexpr(2)
PREC_FP4 = tl.constexpr(3)


@triton.jit
def softplus(dt):
    return tl.where(dt <= 20.0, tl.extra.cuda.libdevice.log1p(tl.exp(dt)), dt)


@triton.jit
def _quantize(
    x,
    u,
    s_g,
    PREC: tl.constexpr,
    SR: tl.constexpr,
    NB16: tl.constexpr,
    NB128: tl.constexpr,
):
    if PREC == PREC_FP32:
        out = x
    elif PREC == PREC_FP16:
        out = quant_tl.quantize_fp16(x, u, SR)
    elif PREC == PREC_FP8:
        out = quant_tl.quantize_blockwise(x, u, SR, NB128, 128, 3, -6, 448.0)
    elif PREC == PREC_FP4:
        out = quant_tl.quantize_nvfp4(x, u, s_g, SR, NB16, 16)
    return out


@triton.jit
def _quantize_rtn(x, s_g, PREC: tl.constexpr, NB16: tl.constexpr, NB128: tl.constexpr):
    return _quantize(x, 0.0, s_g, PREC, False, NB16, NB128)


@triton.jit
def _replay_step_kernel(
    state_hi_ptr,
    state_lo_ptr,
    amax_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    write_pos_ptr,
    is_flush_ptr,
    window_idx_ptr,
    state_idx_ptr,
    null_block_id,
    seed,
    s_x_row,
    s_dt_row,
    s_B_row,
    s_C_row,
    s_out_row,
    s_state_slot,
    s_xc_slot,
    s_dtc_slot,
    s_Bc_slot,
    s_amax_slot,
    R,
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    NPT: tl.constexpr,
    TWOPREC: tl.constexpr,
    PREC_HI: tl.constexpr,
    SR_HI: tl.constexpr,
    PREC_LO: tl.constexpr,
    SG_DEFER: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)
    state_idx = tl.load(state_idx_ptr + pid_r).to(tl.int64)
    if state_idx == null_block_id:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, N)
    offs_l = tl.arange(0, 16)
    mask_m = offs_m < P
    sm = mask_m[:, None]
    pid_g = pid_h // (H // G)
    wp = tl.load(write_pos_ptr + pid_r).to(tl.int64)
    w_idx = tl.load(window_idx_ptr + pid_r)
    is_flush = tl.load(is_flush_ptr + pid_r) != 0
    is_seed = wp == 0 and w_idx == 0
    dt_cur = tl.load(dt_ptr + pid_r * s_dt_row + pid_h).to(tl.float32)
    dt_cur += tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    dt_cur = softplus(dt_cur)
    A = tl.load(A_ptr + pid_h).to(tl.float32)
    x_cur = tl.load(
        x_ptr + pid_r * s_x_row + pid_h * P + offs_m, mask=mask_m, other=0.0
    ).to(tl.float32)
    C = tl.load(C_ptr + pid_r * s_C_row + pid_g * N + offs_n).to(tl.float32)
    B_cur = tl.load(B_ptr + pid_r * s_B_row + pid_g * N + offs_n).to(tl.float32)
    dtc = tl.load(
        dt_cache_ptr + state_idx * s_dtc_slot + pid_h * 16 + offs_l,
        mask=offs_l < wp,
        other=0.0,
    ).to(tl.float32)
    dt_all = tl.where(offs_l == wp, dt_cur, dtc)
    dA_total = A * tl.sum(dt_all, axis=0)
    total_decay = tl.exp(dA_total)
    scale = dt_all * tl.exp(dA_total - A * tl.cumsum(dt_all, axis=0))
    scale = tl.where(offs_l <= wp, scale, 0.0)
    xc = tl.load(
        x_cache_ptr
        + state_idx * s_xc_slot
        + (pid_h * 16 + offs_l[None, :]) * P
        + offs_m[:, None],
        mask=mask_m[:, None] & (offs_l[None, :] < wp),
        other=0.0,
    ).to(tl.float32)
    x_all = tl.where(offs_l[None, :] == wp, x_cur[:, None], xc)
    Bc = tl.load(
        B_cache_ptr
        + state_idx * s_Bc_slot
        + (pid_g * 16 + offs_l[:, None]) * N
        + offs_n[None, :],
        mask=offs_l[:, None] < wp,
        other=0.0,
    ).to(tl.float32)
    B_all = tl.where(offs_l[:, None] == wp, B_cur[None, :], Bc)
    off = state_idx * s_state_slot + (pid_h * P + offs_m[:, None]) * N + offs_n[None, :]
    NB16: tl.constexpr = N // 16
    NB128: tl.constexpr = N // 128
    key = seed + w_idx.to(tl.int32) * 1103515245
    if is_flush:
        B_scaled = (B_all * scale[:, None]).to(tl.bfloat16)
        delta = tl.dot(x_all.to(tl.bfloat16), B_scaled, out_dtype=tl.float32)
        S_new = tl.load(state_hi_ptr + off, mask=sm, other=0.0) * total_decay + delta
        out = tl.sum(S_new * C[None, :], axis=1)
        if SG_DEFER:
            # NVFP4 needs a request-wide reduction after this kernel completes.
            tl.store(state_hi_ptr + off, S_new, mask=sm)
            tl.store(
                amax_ptr + state_idx * s_amax_slot + pid_h * NPT + pid_m,
                tl.max(tl.abs(tl.where(sm, S_new, 0.0))),
            )
        else:
            tl.store(
                state_hi_ptr + off,
                _quantize(S_new, tl.rand(key, off), 0.0, PREC_HI, SR_HI, NB16, NB128),
                mask=sm,
            )
            if TWOPREC:
                # Quantize the unrounded solution independently for the read copy.
                tl.store(
                    state_lo_ptr + off,
                    _quantize_rtn(S_new, 0.0, PREC_LO, NB16, NB128),
                    mask=sm,
                )
    else:
        if TWOPREC and (not is_seed):
            S_read = tl.load(state_lo_ptr + off, mask=sm, other=0.0)
        else:
            S_read = tl.load(state_hi_ptr + off, mask=sm, other=0.0)
        ckpt = tl.sum(S_read * C[None, :], axis=1) * total_decay
        bc = tl.sum(B_all * C[None, :], axis=1)
        cache = tl.sum(x_all * (scale * bc)[None, :], axis=1)
        out = ckpt + cache
        if is_seed:
            if SG_DEFER:
                tl.store(
                    amax_ptr + state_idx * s_amax_slot + pid_h * NPT + pid_m,
                    tl.max(tl.abs(tl.where(sm, S_read, 0.0))),
                )
            else:
                tl.store(
                    state_hi_ptr + off,
                    _quantize(
                        S_read, tl.rand(key, off), 0.0, PREC_HI, SR_HI, NB16, NB128
                    ),
                    mask=sm,
                )
                if TWOPREC:
                    tl.store(
                        state_lo_ptr + off,
                        _quantize_rtn(S_read, 0.0, PREC_LO, NB16, NB128),
                        mask=sm,
                    )
    out += x_cur * tl.load(D_ptr + pid_h).to(tl.float32)
    tl.store(out_ptr + pid_r * s_out_row + pid_h * P + offs_m, out, mask=mask_m)
    if is_flush:
        return
    tl.store(
        x_cache_ptr + state_idx * s_xc_slot + (pid_h * 16 + wp) * P + offs_m,
        x_cur.to(tl.bfloat16),
        mask=mask_m,
    )
    if pid_m == 0:
        tl.store(dt_cache_ptr + state_idx * s_dtc_slot + pid_h * 16 + wp, dt_cur)
        if pid_h % (H // G) == 0:
            tl.store(
                B_cache_ptr + state_idx * s_Bc_slot + (pid_g * 16 + wp) * N + offs_n,
                B_cur.to(tl.bfloat16),
            )


@triton.jit
def _flush_quant_kernel(
    state_hi_ptr,
    state_lo_ptr,
    amax_ptr,
    write_pos_ptr,
    is_flush_ptr,
    window_idx_ptr,
    state_idx_ptr,
    null_block_id,
    seed,
    s_state_slot,
    s_amax_slot,
    R,
    H: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    NPT: tl.constexpr,
    TWOPREC: tl.constexpr,
    PREC_HI: tl.constexpr,
    SR_HI: tl.constexpr,
    PREC_LO: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_h = tl.program_id(2)
    state_idx = tl.load(state_idx_ptr + pid_r).to(tl.int64)
    if state_idx == null_block_id:
        return
    w_idx = tl.load(window_idx_ptr + pid_r)
    is_seed = tl.load(write_pos_ptr + pid_r) == 0 and w_idx == 0
    if tl.load(is_flush_ptr + pid_r) == 0 and (not is_seed):
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, N)
    mask_m = offs_m < P
    sm = mask_m[:, None]
    off = state_idx * s_state_slot + (pid_h * P + offs_m[:, None]) * N + offs_n[None, :]
    S_new = tl.load(state_hi_ptr + off, mask=sm, other=0.0)
    am = 0.0
    # Reduce all heads and row tiles of this request, including zero-padded tiles.
    for h in range(H):
        am = tl.maximum(
            am,
            tl.max(
                tl.load(
                    amax_ptr + state_idx * s_amax_slot + h * NPT + tl.arange(0, NPT)
                )
            ),
        )
    s_g = quant_tl.nvfp4_outer_scale(am)
    NB16: tl.constexpr = N // 16
    NB128: tl.constexpr = N // 128
    key = seed + w_idx.to(tl.int32) * 1103515245
    q_hi = _quantize(S_new, tl.rand(key, off), s_g, PREC_HI, SR_HI, NB16, NB128)
    tl.store(state_hi_ptr + off, q_hi, mask=sm)
    if TWOPREC:
        tl.store(
            state_lo_ptr + off,
            _quantize_rtn(S_new, s_g, PREC_LO, NB16, NB128),
            mask=sm,
        )
