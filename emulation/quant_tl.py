# SPDX-License-Identifier: Apache-2.0
"""Floating-point emulation of FP16, block-128 FP8 and request-scaled NVFP4."""

from __future__ import annotations

import triton
import triton.language as tl

FP16_MANT = tl.constexpr(10)
FP16_MIN_EXP = tl.constexpr(-14)
E4M3_MANT = tl.constexpr(3)
E4M3_MIN_EXP = tl.constexpr(-6)
E4M3_MAX = tl.constexpr(448.0)
E2M1_MANT = tl.constexpr(1)
E2M1_MIN_EXP = tl.constexpr(0)
E2M1_MAX = tl.constexpr(6.0)


@triton.jit
def exp2i(k):
    return (k.to(tl.int32) + 127 << 23).to(tl.float32, bitcast=True)


@triton.jit
def grid_ulp(x, MANT: tl.constexpr, MIN_EXP: tl.constexpr):
    a = tl.abs(x)
    am = tl.maximum(a, 2.0**MIN_EXP)
    e = tl.floor(tl.extra.cuda.libdevice.log2(am))
    e = tl.maximum(e, float(MIN_EXP))
    return tl.where(a < 2.0**MIN_EXP, 2.0 ** (MIN_EXP - MANT), exp2i(e - MANT))


@triton.jit
def cast_grid(x, MANT: tl.constexpr, MIN_EXP: tl.constexpr, MAX_VAL: tl.constexpr):
    xc = tl.minimum(tl.maximum(x, -MAX_VAL), MAX_VAL)
    ulp = grid_ulp(xc, MANT, MIN_EXP)
    return tl.extra.cuda.libdevice.rint(xc / ulp) * ulp


@triton.jit
def sr_grid(x, u, MANT: tl.constexpr, MIN_EXP: tl.constexpr, MAX_VAL: tl.constexpr):
    ulp = grid_ulp(x, MANT, MIN_EXP)
    q = tl.where(x < 0, -1.0, 1.0) * tl.floor(tl.abs(x) / ulp + u) * ulp
    return cast_grid(q, MANT, MIN_EXP, MAX_VAL)


@triton.jit
def rtn_e4m3(x):
    return cast_grid(x, E4M3_MANT, E4M3_MIN_EXP, E4M3_MAX)


@triton.jit
def rtn_fp16(x):
    return x.to(tl.float16).to(tl.float32)


@triton.jit
def quantize_fp16(x, u, SR: tl.constexpr):
    if SR:
        ulp = grid_ulp(x, FP16_MANT, FP16_MIN_EXP)
        out = rtn_fp16(tl.where(x < 0, -1.0, 1.0) * tl.floor(tl.abs(x) / ulp + u) * ulp)
    else:
        out = rtn_fp16(x)
    return out


@triton.jit
def quantize_blockwise(
    x,
    u,
    SR: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    MANT: tl.constexpr,
    MIN_EXP: tl.constexpr,
    MAX_VAL: tl.constexpr,
):
    m: tl.constexpr = x.shape[0]
    xr = tl.reshape(x, (m, NB, BLOCK))
    amax = tl.max(tl.abs(xr), axis=2, keep_dims=True)
    scale = amax / MAX_VAL
    # Guard zero and underflowed scales before division.
    scale = tl.where(scale > 0, scale, 1.0)
    xs = tl.div_rn(xr, scale)
    if SR:
        q = sr_grid(xs, tl.reshape(u, (m, NB, BLOCK)), MANT, MIN_EXP, MAX_VAL)
    else:
        q = cast_grid(xs, MANT, MIN_EXP, MAX_VAL)
    return tl.reshape(q * scale, (m, NB * BLOCK))


@triton.jit
def nvfp4_outer_scale(amax):
    s = amax / (E2M1_MAX * E4M3_MAX)
    return tl.where(s > 0, s, 1.0)


@triton.jit
def quantize_nvfp4(
    x, u, s_g_in, SR: tl.constexpr, NB: tl.constexpr, BLOCK: tl.constexpr
):
    m: tl.constexpr = x.shape[0]
    s_g = s_g_in
    xr = tl.reshape(x, (m, NB, BLOCK))
    amax = tl.max(tl.abs(xr), axis=2, keep_dims=True)
    s_b = rtn_e4m3(tl.div_rn(amax, E2M1_MAX * s_g))
    dead = s_b <= 0.0
    # A zero encoded scale annihilates its block, even under stochastic rounding.
    s_b = tl.where(dead, 1.0, s_b)
    scale = s_b * s_g
    xs = tl.div_rn(xr, scale)
    if SR:
        q = sr_grid(
            xs, tl.reshape(u, (m, NB, BLOCK)), E2M1_MANT, E2M1_MIN_EXP, E2M1_MAX
        )
    else:
        q = cast_grid(xs, E2M1_MANT, E2M1_MIN_EXP, E2M1_MAX)
    out = tl.where(tl.broadcast_to(dead, (m, NB, BLOCK)), 0.0, q * scale)
    return tl.reshape(out, (m, NB * BLOCK))
