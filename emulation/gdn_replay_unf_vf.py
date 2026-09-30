# SPDX-License-Identifier: Apache-2.0
"""GDN replay: cached FP16 updates for outputs, raw inputs for flushes."""

from __future__ import annotations

import triton
import triton.language as tl

from .gdn_common import NORM_EPS, _pick, _row_l2norm
from .mamba2_replay import _quantize, _quantize_rtn, softplus


@triton.jit
def _gdn_unf_vf_step_kernel(
    state_hi_ptr,
    state_lo_ptr,
    amax_ptr,
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    g_cache_ptr,
    beta_cache_ptr,
    u_cache_ptr,
    write_pos_ptr,
    is_flush_ptr,
    window_idx_ptr,
    state_idx_ptr,
    null_block_id,
    seed,
    scale,
    s_qkv_row,
    s_a_row,
    s_b_row,
    s_out_row,
    s_state_slot,
    s_kc_slot,
    s_vc_slot,
    s_gc_slot,
    s_bc_slot,
    s_uc_slot,
    s_amax_slot,
    R,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NVT: tl.constexpr,
    TWOPREC: tl.constexpr,
    PREC_HI: tl.constexpr,
    SR_HI: tl.constexpr,
    PREC_LO: tl.constexpr,
    SG_DEFER: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_hv = tl.program_id(2)
    state_idx = tl.load(state_idx_ptr + pid_r).to(tl.int64)
    if state_idx == null_block_id:
        return
    pid_h = pid_hv // (HV // H)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_k = tl.arange(0, K)
    offs_l = tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_v = offs_v < V
    wp = tl.load(write_pos_ptr + pid_r).to(tl.int64)
    w_idx = tl.load(window_idx_ptr + pid_r)
    is_flush = tl.load(is_flush_ptr + pid_r) != 0
    is_seed = wp == 0 and w_idx == 0
    qkv = mixed_qkv_ptr + pid_r * s_qkv_row
    q_raw = tl.load(qkv + pid_h * K + offs_k).to(tl.float32)
    k_raw = tl.load(qkv + H * K + pid_h * K + offs_k).to(tl.float32)
    v_cur = tl.load(qkv + 2 * H * K + pid_hv * V + offs_v, mask=mask_v, other=0.0).to(
        tl.float32
    )
    a_val = tl.load(a_ptr + pid_r * s_a_row + pid_hv).to(tl.float32)
    b_val = tl.load(b_ptr + pid_r * s_b_row + pid_hv).to(tl.float32)
    A_log = tl.load(A_log_ptr + pid_hv).to(tl.float32)
    dt_b = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
    g_cur = -tl.exp(A_log) * softplus(a_val + dt_b)
    beta_cur = tl.sigmoid(b_val).to(tl.bfloat16).to(tl.float32)
    n_q = tl.sqrt(tl.sum(q_raw * q_raw) + NORM_EPS)
    n_k = tl.sqrt(tl.sum(k_raw * k_raw) + NORM_EPS)
    q_n = q_raw / n_q * scale
    k_n = k_raw / n_k
    kc = tl.load(
        k_cache_ptr
        + state_idx * s_kc_slot
        + (pid_h * 16 + offs_l[:, None]) * K
        + offs_k[None, :],
        mask=offs_l[:, None] < wp,
        other=0.0,
    ).to(tl.float32)
    k_raw_all = tl.where(offs_l[:, None] == wp, k_raw[None, :], kc)
    vc = tl.load(
        v_cache_ptr
        + state_idx * s_vc_slot
        + (pid_hv * 16 + offs_l[:, None]) * V
        + offs_v[None, :],
        mask=(offs_l[:, None] < wp) & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    v_all = tl.where(offs_l[:, None] == wp, v_cur[None, :], vc)
    gc = tl.load(
        g_cache_ptr + state_idx * s_gc_slot + pid_hv * 16 + offs_l,
        mask=offs_l < wp,
        other=0.0,
    ).to(tl.float32)
    g_all = tl.where(offs_l == wp, g_cur, gc)
    bc = tl.load(
        beta_cache_ptr + state_idx * s_bc_slot + pid_hv * 16 + offs_l,
        mask=offs_l < wp,
        other=0.0,
    ).to(tl.float32)
    beta_all = tl.where(offs_l == wp, beta_cur, bc)
    valid = offs_l <= wp
    g_win = tl.where(valid, g_all, 0.0)
    G = tl.cumsum(g_win, axis=0)
    G_t = tl.sum(g_win, axis=0)
    Lam_t = tl.exp(G_t)
    kn = tl.sqrt(tl.sum(k_raw_all * k_raw_all, axis=1) + NORM_EPS)
    kq = tl.sum(k_raw_all * q_n[None, :], axis=1) / kn
    kk_t = tl.sum(k_raw_all * k_n[None, :], axis=1) / kn
    kq_cur = tl.sum(k_n * q_n)
    off = (
        state_idx * s_state_slot + (pid_hv * V + offs_v[:, None]) * K + offs_k[None, :]
    )
    sm = mask_v[:, None]
    if TWOPREC and (not is_flush) and (not is_seed):
        S0 = tl.load(state_lo_ptr + off, mask=sm, other=0.0)
    else:
        S0 = tl.load(state_hi_ptr + off, mask=sm, other=0.0)
    Bop = tl.where(
        offs_n[None, :] == 0,
        q_raw[:, None],
        tl.where(offs_n[None, :] == 1, k_raw[:, None], 0.0),
    )
    C = tl.dot(S0.to(tl.bfloat16), Bop.to(tl.bfloat16), out_dtype=tl.float32)
    c_q = tl.sum(tl.where(offs_n[None, :] == 0, C, 0.0), axis=1) / n_q * scale
    c_k = tl.sum(tl.where(offs_n[None, :] == 1, C, 0.0), axis=1) / n_k
    NB16: tl.constexpr = K // 16
    NB128: tl.constexpr = K // 128
    key = seed + w_idx.to(tl.int32) * 1103515245
    if is_flush:
        # Re-solve from raw k/v; the cached FP16 updates do not enter the anchor.
        k_all = _row_l2norm(k_raw_all, NORM_EPS)
        k_all = k_all.to(tl.bfloat16).to(tl.float32)
        kk = tl.dot(
            k_all.to(tl.bfloat16), tl.trans(k_all).to(tl.bfloat16), out_dtype=tl.float32
        )
        S0K = tl.dot(
            S0.to(tl.bfloat16), tl.trans(k_all).to(tl.bfloat16), out_dtype=tl.float32
        )
        r = v_all - tl.exp(G)[:, None] * tl.trans(S0K)
        r = r.to(tl.bfloat16).to(tl.float32)
        u = tl.zeros([16, BV], dtype=tl.float32)
        for j in range(16):
            Gj = _pick(G, j, offs_l)
            kk_row = tl.sum(tl.where(offs_l[:, None] == j, kk, 0.0), axis=0)
            coef = tl.where(offs_l < j, tl.exp(Gj - G) * kk_row, 0.0)
            acc = tl.sum(u * coef[:, None], axis=0)
            rj = tl.sum(tl.where(offs_l[:, None] == j, r, 0.0), axis=0)
            bj = _pick(beta_all, j, offs_l)
            u = tl.where(offs_l[:, None] == j, (bj * (rj - acc))[None, :], u)
        uw = u * tl.exp(G_t - G)[:, None]
        uw = uw.to(tl.bfloat16).to(tl.float32)
        S_new = tl.exp(G_t) * S0 + tl.dot(
            tl.trans(uw).to(tl.bfloat16), k_all.to(tl.bfloat16), out_dtype=tl.float32
        )
        out = Lam_t * c_q + tl.sum(u * (tl.exp(G_t - G) * kq)[:, None], axis=0)
        tl.store(out_ptr + pid_r * s_out_row + pid_hv * V + offs_v, out, mask=mask_v)
        if SG_DEFER:
            # Pass two forms both quantized copies from this FP32 solution.
            tl.store(state_hi_ptr + off, S_new, mask=sm)
            tl.store(
                amax_ptr + state_idx * s_amax_slot + pid_hv * NVT + pid_v,
                tl.max(tl.abs(tl.where(sm, S_new, 0.0))),
            )
        else:
            tl.store(
                state_hi_ptr + off,
                _quantize(S_new, tl.rand(key, off), 0.0, PREC_HI, SR_HI, NB16, NB128),
                mask=sm,
            )
            if TWOPREC:
                tl.store(
                    state_lo_ptr + off,
                    _quantize_rtn(S_new, 0.0, PREC_LO, NB16, NB128),
                    mask=sm,
                )
        return
    uc = tl.load(
        u_cache_ptr
        + state_idx * s_uc_slot
        + (pid_hv * 16 + offs_l[:, None]) * V
        + offs_v[None, :],
        mask=(offs_l[:, None] < wp) & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    dW = tl.where(offs_l < wp, tl.exp(G_t - G), 0.0)
    Sk = Lam_t * c_k + tl.sum(uc * (dW * kk_t)[:, None], axis=0)
    u_t = beta_cur * (v_cur - Sk)
    out = Lam_t * c_q + tl.sum(uc * (dW * kq)[:, None], axis=0) + u_t * kq_cur
    tl.store(out_ptr + pid_r * s_out_row + pid_hv * V + offs_v, out, mask=mask_v)
    if is_seed:
        if SG_DEFER:
            tl.store(
                amax_ptr + state_idx * s_amax_slot + pid_hv * NVT + pid_v,
                tl.max(tl.abs(tl.where(sm, S0, 0.0))),
            )
        else:
            tl.store(
                state_hi_ptr + off,
                _quantize(S0, tl.rand(key, off), 0.0, PREC_HI, SR_HI, NB16, NB128),
                mask=sm,
            )
            if TWOPREC:
                tl.store(
                    state_lo_ptr + off,
                    _quantize_rtn(S0, 0.0, PREC_LO, NB16, NB128),
                    mask=sm,
                )
    tl.store(
        u_cache_ptr + state_idx * s_uc_slot + (pid_hv * 16 + wp) * V + offs_v,
        u_t.to(tl.float16),
        mask=mask_v,
    )
    tl.store(
        v_cache_ptr + state_idx * s_vc_slot + (pid_hv * 16 + wp) * V + offs_v,
        v_cur.to(tl.bfloat16),
        mask=mask_v,
    )
    if pid_v == 0:
        tl.store(g_cache_ptr + state_idx * s_gc_slot + pid_hv * 16 + wp, g_cur)
        tl.store(beta_cache_ptr + state_idx * s_bc_slot + pid_hv * 16 + wp, beta_cur)
        if pid_hv % (HV // H) == 0:
            tl.store(
                k_cache_ptr + state_idx * s_kc_slot + (pid_h * 16 + wp) * K + offs_k,
                k_raw.to(tl.bfloat16),
            )
