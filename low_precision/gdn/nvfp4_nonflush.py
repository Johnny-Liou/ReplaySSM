# SPDX-License-Identifier: Apache-2.0
"""GDN nvfp4 nonflush; fixed L=16 specialization."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float16, Float32, Int8, Int32, Int64, Uint32, Uint64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from low_precision.common.nf_ops import (
    bf16x4_bits_to_f32x4,
    cp_async_bulk_g2s,
    ldg_nc_v4_u32,
    lds_e4m3x2_to_f32x2,
    lds_v4_u32,
)


@dsl_user_op
def lds_f16_f32(smem_addr, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 v;\n\tld.shared.u16 v, [$1];\n\tcvt.f32.f16 $0, v;\n\t}\n",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def bf16x2_bits_to_f16x2(bits: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [Uint32(bits).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .b16 lo, hi;\n"
            "\t.reg .f32 a, b;\n"
            "\tmov.b32 {lo, hi}, $1;\n"
            "\tmov.b32 a, {0, lo};\n"
            "\tmov.b32 b, {0, hi};\n"
            "\tcvt.rn.f16x2.f32 $0, b, a;\n"
            "\t}\n"
        ),
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


def _dot2_asm() -> str:
    lines = [
        "{",
        ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;",
        ".reg .b32 h, accq, acck;",
        ".reg .b16 lo, hi;",
        ".reg .f32 flo, fhi;",
        "mov.b32 {b0, b1, b2, b3}, $2;",
        "mov.b32 {b4, b5, b6, b7}, $3;",
        "mov.b32 accq, 0;",
        "mov.b32 acck, 0;",
    ]
    for i in range(8):
        lines += [
            f"cvt.rn.f16x2.e2m1x2 h, b{i};",
            f"fma.rn.f16x2 accq, h, ${4 + i}, accq;",
            f"fma.rn.f16x2 acck, h, ${12 + i}, acck;",
        ]
    lines += [
        "mov.b32 {lo, hi}, accq;",
        "cvt.f32.f16 flo, lo;",
        "cvt.f32.f16 fhi, hi;",
        "add.f32 $0, flo, fhi;",
        "mov.b32 {lo, hi}, acck;",
        "cvt.f32.f16 flo, lo;",
        "cvt.f32.f16 fhi, hi;",
        "add.f32 $1, flo, fhi;",
        "}",
    ]
    return "\n\t".join(lines) + "\n"


_DOT2_ASM = _dot2_asm()


@dsl_user_op
def e2m1_bytes8_dot2_f16x2(
    w0,
    w1,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    k0,
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    k7,
    *,
    loc=None,
    ip=None,
):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1)]
        + [
            Uint32(x).ir_value(loc=loc, ip=ip)
            for x in (q0, q1, q2, q3, q4, q5, q6, q7, k0, k1, k2, k3, k4, k5, k6, k7)
        ],
        _DOT2_ASM,
        "=f,=f,r,r," + ",".join(["r"] * 16),
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    dq = Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip))
    dk = Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip))
    return (dq, dk)


class Sm103GdnTp4NonflushFused:
    def __init__(
        self, h: int, hv: int, k_dim: int = 128, v_dim: int = 128, cache_len: int = 16
    ) -> None:
        assert k_dim == 128 and v_dim == 128 and (cache_len == 16)
        assert hv % h == 0
        self.gqa = hv // h
        self.nstage = 3
        self.min_blocks = 5
        assert self.nstage <= self.gqa <= 4
        self.h, self.hv = (h, hv)
        self.k_dim, self.v_dim, self.cache_len = (k_dim, v_dim, cache_len)
        self.num_warps = 4
        self.row_bytes = k_dim // 2
        self.head_bytes = v_dim * self.row_bytes
        self.head_sf_bytes = 1024
        self.head_u_bytes = cache_len * v_dim * 2
        self.stage_tx = self.head_bytes + self.head_sf_bytes + self.head_u_bytes

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        u_cache: cute.Tensor,
        v_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        state_lo: cute.Tensor,
        sf_lo: cute.Tensor,
        sg_meta: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        stream: CUstream,
    ):
        self.kernel(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            u_cache,
            v_cache,
            k_cache,
            g_cache,
            beta_cache,
            state_lo,
            sf_lo,
            sg_meta,
            state_indices,
            write_pos,
            is_flush,
            out,
            scale,
        ).launch(
            grid=(q.shape[0], self.h, 1),
            block=(self.num_warps * 32, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        u_cache: cute.Tensor,
        v_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        state_lo: cute.Tensor,
        sf_lo: cute.Tensor,
        sg_meta: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        batch_id, group, _ = cute.arch.block_idx()
        L = self.cache_len
        gqa = self.gqa
        smem = cutlass.utils.SmemAllocator()
        s_state = smem.allocate_tensor(
            Int8, cute.make_layout(3 * self.head_bytes), byte_alignment=128
        )
        s_sf = smem.allocate_tensor(
            Int8, cute.make_layout(3 * self.head_sf_bytes), byte_alignment=128
        )
        s_u = smem.allocate_tensor(
            Int8, cute.make_layout(3 * self.head_u_bytes), byte_alignment=128
        )
        s_q = smem.allocate_array(Float32, self.k_dim)
        s_k = smem.allocate_array(Float32, self.k_dim)
        s_ckq = smem.allocate_array(Float32, self.v_dim)
        s_ckk = smem.allocate_array(Float32, self.v_dim)
        s_wq = smem.allocate_array(Float32, gqa * L)
        s_wk = smem.allocate_array(Float32, gqa * L)
        s_kq = smem.allocate_array(Float32, L)
        s_kk = smem.allocate_array(Float32, L)
        s_lam = smem.allocate_array(Float32, gqa)
        s_beta = smem.allocate_array(Float32, gqa)
        s_red = smem.allocate_array(Float32, 16)
        bars = smem.allocate_array(Int64, self.nstage)
        flag = Int32(is_flush[batch_id])
        warp = tid // 32
        lane = tid % 32
        subgroup = lane // 4
        lig = lane % 4
        first_head = group * gqa
        state_idx = state_indices[batch_id]
        wp = Int32(write_pos[batch_id])
        sg_state = Float32(sg_meta[state_idx, 2, group])
        sf_lane = 2 * lig % 4 + 2 * lig // 4 * 512
        if flag == 0:
            if state_idx <= 0:
                for p in cutlass.range_constexpr(gqa):
                    out[batch_id, first_head + p, tid] = BFloat16(0.0)
            else:
                head_base = Int64(state_idx * self.hv + first_head)
                state_g = state_lo.iterator.toint() + head_base * Int64(self.head_bytes)
                sf_g = sf_lo.iterator.toint() + head_base * Int64(self.head_sf_bytes)
                u_g = u_cache.iterator.toint() + head_base * Int64(self.head_u_bytes)
                state_s = s_state.iterator.toint()
                sf_s = s_sf.iterator.toint()
                u_s = s_u.iterator.toint()
                bar_s = bars.toint()
                if tid == 0:
                    for s in cutlass.range_constexpr(self.nstage):
                        cute.arch.mbarrier_init(bars + s, 1)
                    cute.arch.mbarrier_init_fence()
                    for s in cutlass.range_constexpr(self.nstage - 1):
                        if s < gqa:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                bars + s, self.stage_tx
                            )
                            cp_async_bulk_g2s(
                                state_s + s * self.head_bytes,
                                state_g + s * self.head_bytes,
                                self.head_bytes,
                                bar_s + s * 8,
                            )
                            cp_async_bulk_g2s(
                                sf_s + s * self.head_sf_bytes,
                                sf_g + s * self.head_sf_bytes,
                                self.head_sf_bytes,
                                bar_s + s * 8,
                            )
                            cp_async_bulk_g2s(
                                u_s + s * self.head_u_bytes,
                                u_g + s * self.head_u_bytes,
                                self.head_u_bytes,
                                bar_s + s * 8,
                            )
                q_val = Float32(q[batch_id, group, tid])
                k_val = Float32(k[batch_id, group, tid])
                k_u64 = cute.recast_tensor(k_cache, Uint64)
                kq_words = cute.make_rmem_tensor((4,), Uint64)
                for j in cutlass.range_constexpr(4):
                    kq_words[j] = k_u64[state_idx, group, warp + 4 * j, lane]
                qk_row = Int64(
                    (batch_id * self.h + group) * self.k_dim + lig * 32
                ) * Int64(2)
                q_g = q.iterator.toint() + qk_row
                k_g = k.iterator.toint() + qk_row
                qw = cute.make_rmem_tensor((16,), Uint32)
                kw = cute.make_rmem_tensor((16,), Uint32)
                for i in cutlass.range_constexpr(4):
                    a0, a1, a2, a3 = ldg_nc_v4_u32(q_g + i * 16)
                    qw[4 * i] = bf16x2_bits_to_f16x2(a0)
                    qw[4 * i + 1] = bf16x2_bits_to_f16x2(a1)
                    qw[4 * i + 2] = bf16x2_bits_to_f16x2(a2)
                    qw[4 * i + 3] = bf16x2_bits_to_f16x2(a3)
                    b0, b1, b2, b3 = ldg_nc_v4_u32(k_g + i * 16)
                    kw[4 * i] = bf16x2_bits_to_f16x2(b0)
                    kw[4 * i + 1] = bf16x2_bits_to_f16x2(b1)
                    kw[4 * i + 2] = bf16x2_bits_to_f16x2(b2)
                    kw[4 * i + 3] = bf16x2_bits_to_f16x2(b3)
                v_next = v[batch_id, first_head, tid]
                v_next2 = v_next
                if cutlass.const_expr(gqa > 1):
                    v_next2 = v[batch_id, first_head + 1, tid]
                g_ring = Float32(0.0)
                a_h = Float32(0.0)
                b_h = Float32(0.0)
                alog_h = Float32(0.0)
                dtb_h = Float32(0.0)
                if warp < gqa:
                    hp = first_head + warp
                    if lane < L:
                        g_ring = g_cache[state_idx, hp, lane]
                    a_h = Float32(a[batch_id, hp])
                    b_h = Float32(b[batch_id, hp])
                    alog_h = Float32(a_log[hp])
                    dtb_h = Float32(dt_bias[hp])
                s_q[tid] = q_val
                s_k[tid] = k_val
                pq = cute.arch.warp_reduction_sum(q_val * q_val)
                pk = cute.arch.warp_reduction_sum(k_val * k_val)
                pqk = cute.arch.warp_reduction_sum(q_val * k_val)
                if lane == 0:
                    s_red[warp * 3] = pq
                    s_red[warp * 3 + 1] = pk
                    s_red[warp * 3 + 2] = pqk
                k_cache[state_idx, group, wp, tid] = k[batch_id, group, tid]
                cute.arch.sync_threads()
                sq2 = s_red[0] + s_red[3] + s_red[6] + s_red[9]
                sk2 = s_red[1] + s_red[4] + s_red[7] + s_red[10]
                sqk = s_red[2] + s_red[5] + s_red[8] + s_red[11]
                n_q = cute.math.sqrt(sq2 + Float32(1e-06))
                n_k = cute.math.sqrt(sk2 + Float32(1e-06))
                inv_nq_scale = scale / n_q
                inv_nk = Float32(1.0) / n_k
                kq_cur = sqk * inv_nq_scale * inv_nk
                for head in cutlass.range(gqa, unroll=1):
                    stage = head % self.nstage
                    phase = head // self.nstage % 2
                    nxt = head + self.nstage - 1
                    if nxt < gqa:
                        if tid == 0:
                            nst = nxt % self.nstage
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                bars + nst, self.stage_tx
                            )
                            cp_async_bulk_g2s(
                                state_s + nst * self.head_bytes,
                                state_g + nxt * self.head_bytes,
                                self.head_bytes,
                                bar_s + nst * 8,
                            )
                            cp_async_bulk_g2s(
                                sf_s + nst * self.head_sf_bytes,
                                sf_g + nxt * self.head_sf_bytes,
                                self.head_sf_bytes,
                                bar_s + nst * 8,
                            )
                            cp_async_bulk_g2s(
                                u_s + nst * self.head_u_bytes,
                                u_g + nxt * self.head_u_bytes,
                                self.head_u_bytes,
                                bar_s + nst * 8,
                            )
                    hp = first_head + head
                    v_val = v_next
                    v_next = v_next2
                    if head + 2 < gqa:
                        v_next2 = v[batch_id, hp + 2, tid]
                    cute.arch.mbarrier_wait(bars + stage, phase)
                    tile_s = state_s + stage * self.head_bytes
                    tile_sf_s = sf_s + stage * self.head_sf_bytes
                    for rr in cutlass.range_constexpr(4):
                        row = rr * 32 + warp * 8 + subgroup
                        w0, w1, w2, w3 = lds_v4_u32(
                            tile_s + row * self.row_bytes + lig * 16
                        )
                        sf_off = row % 32 * 16 + row // 32 % 4 * 4 + sf_lane
                        sa, sb = lds_e4m3x2_to_f32x2(tile_sf_s + sf_off)
                        sa = sa * sg_state
                        sb = sb * sg_state
                        dq0, dk0 = e2m1_bytes8_dot2_f16x2(
                            w0,
                            w1,
                            qw[0],
                            qw[1],
                            qw[2],
                            qw[3],
                            qw[4],
                            qw[5],
                            qw[6],
                            qw[7],
                            kw[0],
                            kw[1],
                            kw[2],
                            kw[3],
                            kw[4],
                            kw[5],
                            kw[6],
                            kw[7],
                        )
                        dq1, dk1 = e2m1_bytes8_dot2_f16x2(
                            w2,
                            w3,
                            qw[8],
                            qw[9],
                            qw[10],
                            qw[11],
                            qw[12],
                            qw[13],
                            qw[14],
                            qw[15],
                            kw[8],
                            kw[9],
                            kw[10],
                            kw[11],
                            kw[12],
                            kw[13],
                            kw[14],
                            kw[15],
                        )
                        pq_row = dq0 * sa + dq1 * sb
                        pk_row = dk0 * sa + dk1 * sb
                        pq_row = cute.arch.warp_reduction_sum(
                            pq_row, threads_in_group=4
                        )
                        pk_row = cute.arch.warp_reduction_sum(
                            pk_row, threads_in_group=4
                        )
                        if lig == 0:
                            s_ckq[row] = pq_row
                            s_ckk[row] = pk_row
                    if head == 0:
                        for j in cutlass.range_constexpr(4):
                            c = warp + 4 * j
                            x0, x1, x2, x3 = bf16x4_bits_to_f32x4(kq_words[j])
                            pq_c = (
                                x0 * s_q[4 * lane]
                                + x1 * s_q[4 * lane + 1]
                                + x2 * s_q[4 * lane + 2]
                                + x3 * s_q[4 * lane + 3]
                            )
                            pk_c = (
                                x0 * s_k[4 * lane]
                                + x1 * s_k[4 * lane + 1]
                                + x2 * s_k[4 * lane + 2]
                                + x3 * s_k[4 * lane + 3]
                            )
                            pn_c = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
                            pq_c = cute.arch.warp_reduction_sum(pq_c)
                            pk_c = cute.arch.warp_reduction_sum(pk_c)
                            pn_c = cute.arch.warp_reduction_sum(pn_c)
                            if lane == 0:
                                rn = cute.math.rsqrt(pn_c + Float32(1e-06))
                                s_kq[c] = pq_c * rn * inv_nq_scale
                                s_kk[c] = pk_c * rn * inv_nk
                        cute.arch.sync_threads()
                        if warp < gqa:
                            hp0 = first_head + warp
                            sp = a_h + dtb_h
                            if sp <= Float32(20.0):
                                sp = cute.math.log(
                                    Float32(1.0) + cute.math.exp(sp, fastmath=True),
                                    fastmath=True,
                                )
                            g_t = -cute.math.exp(alog_h, fastmath=True) * sp
                            beta_t = Float32(
                                BFloat16(
                                    Float32(1.0)
                                    / (
                                        Float32(1.0)
                                        + cute.math.exp(-b_h, fastmath=True)
                                    )
                                )
                            )
                            c = lane % L
                            gv = Float32(0.0)
                            if c < wp:
                                gv = g_ring
                            if c == wp:
                                gv = g_t
                            prefix = gv
                            for offset in (1, 2, 4, 8):
                                lower = cute.arch.shuffle_sync_up(
                                    prefix, offset, mask_and_clamp=0
                                )
                                if c >= offset:
                                    prefix += lower
                            total = cute.arch.shuffle_sync(prefix, 15)
                            wgt = Float32(0.0)
                            if c < wp:
                                wgt = cute.math.exp(total - prefix, fastmath=True)
                            if lane < L:
                                s_wq[warp * L + c] = wgt * s_kq[c]
                                s_wk[warp * L + c] = wgt * s_kk[c]
                            if lane == 0:
                                s_lam[warp] = cute.math.exp(total, fastmath=True)
                                s_beta[warp] = beta_t
                                g_cache[state_idx, hp0, wp] = g_t
                                beta_cache[state_idx, hp0, wp] = beta_t
                    cute.arch.sync_threads()
                    u_row_s = u_s + stage * self.head_u_bytes + tid * 2
                    sk_hist = Float32(0.0)
                    o_hist = Float32(0.0)
                    for c in cutlass.range_constexpr(L):
                        if c < wp:
                            uv = lds_f16_f32(u_row_s + c * (self.v_dim * 2))
                            sk_hist += uv * s_wk[head * L + c]
                            o_hist += uv * s_wq[head * L + c]
                    lam = s_lam[head]
                    c_q = s_ckq[tid] * inv_nq_scale
                    c_k = s_ckk[tid] * inv_nk
                    vf = Float32(v_val)
                    u_t = s_beta[head] * (vf - (lam * c_k + sk_hist))
                    o = lam * c_q + o_hist + u_t * kq_cur
                    out[batch_id, hp, tid] = BFloat16(o)
                    u_cache[state_idx, hp, wp, tid] = Float16(u_t)
                    v_cache[state_idx, hp, wp, tid] = v_val
                    cute.arch.sync_threads()

    @cache
    @staticmethod
    def compile(
        h: int, hv: int, k_dim: int = 128, v_dim: int = 128, cache_len: int = 16
    ):
        batch = cute.sym_int()
        slots = cute.sym_int()
        q = make_fake_tensor(BFloat16, (batch, h, k_dim), divisibility=16)
        k = make_fake_tensor(BFloat16, (batch, h, k_dim), divisibility=16)
        v = make_fake_tensor(BFloat16, (batch, hv, v_dim), divisibility=16)
        a = make_fake_tensor(BFloat16, (batch, hv), divisibility=16)
        b = make_fake_tensor(BFloat16, (batch, hv), divisibility=16)
        a_log = make_fake_tensor(Float32, (hv,), divisibility=16)
        dt_bias = make_fake_tensor(Float32, (hv,), divisibility=16)
        u_cache = make_fake_tensor(
            Float16, (slots, hv, cache_len, v_dim), divisibility=16
        )
        v_cache = make_fake_tensor(
            BFloat16, (slots, hv, cache_len, v_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            BFloat16, (slots, h, cache_len, k_dim), divisibility=16
        )
        g_cache = make_fake_tensor(Float32, (slots, hv, cache_len), divisibility=16)
        beta_cache = make_fake_tensor(Float32, (slots, hv, cache_len), divisibility=16)
        state_lo = make_fake_tensor(
            Int8, (slots, hv, v_dim, k_dim // 2), divisibility=16
        )
        sf_lo = make_fake_tensor(Int8, (slots, hv, 4, 256), divisibility=16)
        sg_meta = make_fake_tensor(Float32, (slots, 3, h), divisibility=1)
        state_indices = make_fake_tensor(Int32, (batch,), divisibility=1)
        write_pos = make_fake_tensor(Int32, (batch,), divisibility=1)
        is_flush = make_fake_tensor(Int8, (batch,), divisibility=1)
        out = make_fake_tensor(BFloat16, (batch, hv, v_dim), divisibility=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm103GdnTp4NonflushFused(h, hv, k_dim, v_dim, cache_len),
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            u_cache,
            v_cache,
            k_cache,
            g_cache,
            beta_cache,
            state_lo,
            sf_lo,
            sg_meta,
            state_indices,
            write_pos,
            is_flush,
            out,
            Float32(1.0),
            stream,
            options="--enable-tvm-ffi",
        )


def gdn_nvfp4_nonflush(
    q,
    k,
    v,
    a,
    b,
    a_log,
    dt_bias,
    scale,
    u_cache,
    v_cache,
    k_cache,
    g_cache,
    beta_cache,
    state_lo,
    sf_lo,
    sg_meta,
    state_indices,
    write_pos,
    is_flush,
    out,
):
    _, h, k_dim = q.shape
    _, hv, v_dim = v.shape
    slots = state_lo.shape[0]
    cache_len = u_cache.shape[2]
    if write_pos.dtype != torch.int32:
        raise TypeError("the non-flush kernel takes int32 write positions")
    Sm103GdnTp4NonflushFused.compile(h, hv, k_dim, v_dim, cache_len)(
        q,
        k,
        v,
        a,
        b,
        a_log.float(),
        dt_bias.float(),
        u_cache,
        v_cache,
        k_cache,
        g_cache,
        beta_cache,
        state_lo.view(torch.int8),
        sf_lo.view(torch.int8).view(slots, hv, 4, 256),
        sg_meta,
        state_indices,
        write_pos,
        is_flush.view(torch.int8),
        out,
        float(scale),
    )
    return out
