# SPDX-License-Identifier: Apache-2.0
"""GDN nvfp4 flush; fixed L=16 specialization."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
    Float16,
    Float32,
    Int8,
    Int32,
    Int64,
    Uint32,
    Uint64,
    cute,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from low_precision.common import _tcgen05, simple_tma_copy
from low_precision.common.cvt_fp4 import f32_to_e4m3_byte, f32x8_to_e2m1x8
from low_precision.common.gdn_ops import (
    _alloc_tmem,
    _cvt_f32x2_f16x2_rs,
    _dealloc_tmem,
    _make_fp16_idesc,
    _make_sdesc_32B_swizzle,
    _philox4x32,
)

E2M1_MAX = 6.0
E4M3_MAX = 448.0


@dsl_user_op
def e4m3_byte_to_f32(byte: Uint32, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Uint32(byte).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .b16 h, lo, hi;\n"
            "\t.reg .b32 p;\n"
            "\tcvt.u16.u32 h, $1;\n"
            "\tcvt.rn.f16x2.e4m3x2 p, h;\n"
            "\tmov.b32 {lo, hi}, p;\n"
            "\tcvt.f32.f16 $0, lo;\n"
            "\t}\n"
        ),
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def st_global_v2_u32(gaddr, a: Uint32, b: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(gaddr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_u32(gaddr, a: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(gaddr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def rcp_rn_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "rcp.rn.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


class Sm103GdnTp4FlushKernel:
    def __init__(
        self, h: int, hv: int, k_dim: int = 128, v_dim: int = 128, cache_len: int = 16
    ) -> None:
        assert k_dim == 128 and v_dim == 128 and (cache_len == 16)
        assert hv % h == 0 and 1 <= hv // h <= 4
        self.h, self.hv, self.k_dim, self.v_dim, self.cache_len = (
            h,
            hv,
            k_dim,
            v_dim,
            cache_len,
        )
        self.gqa = hv // h
        self.dot_k = 16
        self.num_warps = 6
        self.mma_n = 16
        self.elems = 64
        self.n_groups = 2
        self.unit = 8

    @cute.jit
    def _make_state_tma(self, state: cute.Tensor, op):
        num_elems = 64
        swizzle = cute.make_swizzle(3, 4, 3)
        smem_layout = cute.make_layout(
            (1, 1, self.v_dim, (num_elems, 1), 1),
            stride=(
                0,
                0,
                num_elems,
                (1, self.v_dim * num_elems),
                self.v_dim * num_elems,
            ),
        )
        smem_layout = cute.make_composed_layout(swizzle, 0, smem_layout)
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(state, (None, None, None, num_elems)),
            smem_layout,
            cta_tiler=(1, 1, self.v_dim, num_elems),
        )

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
        state: cute.Tensor,
        v_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        s_lo: cute.Tensor,
        sf_lo: cute.Tensor,
        sg_meta: cute.Tensor,
        sg_parity: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        sr_key0: Uint32,
        sr_key1: Uint32,
        stream: CUstream,
    ):
        state_tma = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        state_tma_store = self._make_state_tma(state, cpasync.CopyBulkTensorTileS2GOp())
        self.kernel(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            v_cache,
            k_cache,
            g_cache,
            beta_cache,
            s_lo,
            sf_lo,
            sg_meta,
            sg_parity,
            state_indices,
            write_pos,
            is_flush,
            out,
            scale,
            state_tma,
            state_tma_store,
            sr_key0,
            sr_key1,
        ).launch(
            grid=(q.shape[0], self.h, 1),
            block=(self.num_warps * 32, 1, 1),
            min_blocks_per_mp=4,
            stream=stream,
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
        v_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        s_lo: cute.Tensor,
        sf_lo: cute.Tensor,
        sg_meta: cute.Tensor,
        sg_parity: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        state_tma: cpasync.TmaInfo,
        state_tma_store: cpasync.TmaInfo,
        sr_key0: Uint32,
        sr_key1: Uint32,
    ):
        L = self.cache_len
        gqa = self.gqa
        elems = self.elems
        uw = self.unit
        n_slots = gqa * 2
        tid, _, _ = cute.arch.thread_idx()
        batch_id, key_head, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        first_head = key_head * gqa
        smem = cutlass.utils.SmemAllocator()

        def allocate_tensor(dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        state_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (self.v_dim, (elems, 2)), stride=(elems, (1, self.v_dim * elems))
            ),
        )
        s_state = allocate_tensor(Float16, state_layout)
        state_unit_layout = cute.make_layout(
            (self.v_dim, ((uw, elems // uw), 2)),
            stride=(elems, ((1, uw), self.v_dim * elems)),
        )
        s_state_u = cute.make_tensor(s_state.iterator, state_unit_layout)
        qk_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (self.mma_n, (elems, 2)), stride=(elems, (1, self.mma_n * elems))
            ),
        )
        s_kq16 = allocate_tensor(Float16, qk_layout)
        dot_layout = cute.make_layout(
            (self.v_dim, (self.dot_k, 2)),
            stride=(self.dot_k, (1, self.v_dim * self.dot_k)),
        )
        s_dk_raw = smem.allocate_tensor(
            Float16, cute.make_layout(2 * self.v_dim * self.dot_k), byte_alignment=128
        )
        dk_base = cute.recast_ptr(
            s_dk_raw.iterator, cute.make_swizzle(1, 4, 3), dtype=Float16
        )
        s_dk = cute.make_tensor(dk_base, dot_layout)
        dot_unit_layout = cute.make_layout(
            (self.v_dim, ((8, self.dot_k // 8), 2)),
            stride=(self.dot_k, ((1, 8), self.v_dim * self.dot_k)),
        )
        s_dk_u = cute.make_tensor(dk_base, dot_unit_layout)
        s_khat32 = smem.allocate_array(Float32, L * self.k_dim)
        s_gram = smem.allocate_array(Float32, L * L)
        s_coef = smem.allocate_array(Float32, gqa * L * L)
        s_qhat = smem.allocate_array(Float32, self.k_dim)
        s_norm_partial = smem.allocate_array(Float32, 4 + 4 * L)
        s_rn = smem.allocate_array(Float32, L)
        s_G = smem.allocate_array(Float32, gqa * L)
        s_expG = smem.allocate_array(Float32, gqa * L)
        s_dw = smem.allocate_array(Float32, gqa * L)
        s_bj = smem.allocate_array(Float32, gqa * L)
        s_lamh = smem.allocate_array(Float32, gqa)
        s_misc = smem.allocate_array(Float32, 8)
        s_gmax = smem.allocate_array(Float32, 4)
        f_state = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        f_stepa = [smem.allocate_array(Int64, 1) for _ in range(gqa)]
        f_gdone = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        fd = [smem.allocate_array(Int64, 1) for _ in range(gqa)]
        f_mma_a = [smem.allocate_array(Int64, 1) for _ in range(gqa)]
        f_mma_b = [smem.allocate_array(Int64, 1) for _ in range(gqa)]
        f_epi_a = [smem.allocate_array(Int64, 1) for _ in range(gqa)]
        k16_full = smem.allocate_array(Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)
        state_idx = state_indices[batch_id]
        flag = Int32(is_flush[batch_id])
        if flag != 0 and state_idx > 0:
            position_tag = write_pos[batch_id]
            tag_u64 = Uint64(position_tag)
            wp = Int32(tag_u64 & Uint64(L - 1))
            event = tag_u64 >> Uint64(4)
            event_lo = Uint32(event)
            event_hi = Uint32(event >> Uint64(32))
            if warp_id == 0:
                with cute.arch.elect_one():
                    for s in cutlass.range_constexpr(n_slots):
                        cute.arch.mbarrier_init(f_state[s], 1)
                        cute.arch.mbarrier_init(f_gdone[s], 128)
                    for p in cutlass.range_constexpr(gqa):
                        cute.arch.mbarrier_init(f_stepa[p], 1)
                        cute.arch.mbarrier_init(fd[p], 128)
                        cute.arch.mbarrier_init(f_mma_a[p], 1)
                        cute.arch.mbarrier_init(f_mma_b[p], 1)
                        cute.arch.mbarrier_init(f_epi_a[p], 128)
                    cute.arch.mbarrier_init(k16_full, 128)
                    cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            if warp_id == 0:
                cpasync.prefetch_descriptor(state_tma.atom)
                cpasync.prefetch_descriptor(state_tma_store.atom)
                for p in cutlass.range_constexpr(gqa):
                    value_head = first_head + p
                    for g in cutlass.range_constexpr(2):
                        slot = p * 2 + g
                        if p > 0:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        g_state = state_tma.tma_tensor[
                            state_idx, value_head, None, (None, g)
                        ]
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                f_state[slot], 16384
                            )
                        simple_tma_copy(
                            state_tma.atom,
                            g_state,
                            s_state[None, (None, g)],
                            f_state[slot],
                        )
                    for g in cutlass.range_constexpr(2):
                        slot = p * 2 + g
                        cute.arch.mbarrier_wait(f_gdone[slot], 0)
                        g_out = state_tma_store.tma_tensor[
                            state_idx, value_head, None, (None, g)
                        ]
                        simple_tma_copy(
                            state_tma_store.atom, s_state[None, (None, g)], g_out
                        )
                        cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
            elif warp_id == 1:
                _alloc_tmem(tmem_addr, 64)
                cute.arch.relinquish_tmem_alloc_permit()
                tmem_base = tmem_addr[0]
                sdesc = _tcgen05.make_sdesc_128B_swizzle(self.v_dim * 128)
                kq_base = sdesc | s_kq16.iterator.toint() >> 4
                state_base0 = sdesc | s_state.iterator.toint() >> 4
                state_base1 = sdesc | s_state.iterator.toint() + 16384 >> 4
                idesc16 = _make_fp16_idesc(self.v_dim, self.mma_n)
                s_d = s_dk[None, (None, 0)]
                s_k16 = s_dk[None, (None, 1)]
                d_base = _make_sdesc_32B_swizzle(Uint64(s_d.iterator.toint() >> 4))
                k_base = _make_sdesc_32B_swizzle(Uint64(s_k16.iterator.toint() >> 4))
                idesc64 = _make_fp16_idesc(self.v_dim, 64)
                cute.arch.mbarrier_wait(k16_full, 0)
                for p in cutlass.range_constexpr(gqa):
                    for g in cutlass.range_constexpr(2):
                        cute.arch.mbarrier_wait(f_state[p * 2 + g], 0)
                        _tcgen05.fence_after_thread_sync()
                        stage_base = state_base0 if g == 0 else state_base1
                        for k_atom in cutlass.range_constexpr(4):
                            state_desc = stage_base | k_atom * 32 >> 4
                            kq_desc = kq_base | g * self.mma_n * 128 + k_atom * 32 >> 4
                            _tcgen05.mma_f16(
                                tmem_base,
                                state_desc,
                                kq_desc,
                                idesc16,
                                g > 0 or k_atom > 0,
                            )
                    _tcgen05.commit(f_stepa[p])
                    cute.arch.mbarrier_wait(fd[p], 0)
                    _tcgen05.fence_after_thread_sync()
                    _tcgen05.mma_f16(tmem_base, d_base, k_base, idesc64, False)
                    _tcgen05.commit(f_mma_a[p])
                    cute.arch.mbarrier_wait(f_epi_a[p], 0)
                    _tcgen05.fence_after_thread_sync()
                    _tcgen05.mma_f16(
                        tmem_base,
                        d_base,
                        k_base + (64 * 2 * self.dot_k >> 4),
                        idesc64,
                        False,
                    )
                    _tcgen05.commit(f_mma_b[p])
                cute.arch.mbarrier_wait(f_gdone[n_slots - 1], 0)
                _dealloc_tmem(tmem_addr[0], 64)
            else:
                consumer_warp = warp_id - 2
                k_idx = consumer_warp * 32 + lane_id
                tmem_row = k_idx
                row = tmem_row ^ 64
                q_raw = Float32(q[batch_id, key_head, k_idx])
                k_raw = Float32(k[batch_id, key_head, k_idx])
                q_sq = cute.arch.warp_reduction_sum(q_raw * q_raw)
                if lane_id == 0:
                    s_norm_partial[consumer_warp] = q_sq
                kwin = cute.make_rmem_tensor((L,), Float32)
                for j in cutlass.range_constexpr(L):
                    kwin[j] = Float32(k_cache[state_idx, key_head, j, k_idx])
                for j in cutlass.range_constexpr(L):
                    kj = kwin[j]
                    if j >= wp:
                        kj = Float32(0.0)
                    if j == wp:
                        kj = k_raw
                    kwin[j] = kj
                    pj = cute.arch.warp_reduction_sum(kj * kj)
                    if lane_id == 0:
                        s_norm_partial[4 + consumer_warp * L + j] = pj
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                if consumer_warp == 0 and lane_id < L:
                    tot = Float32(0.0)
                    for w in cutlass.range_constexpr(4):
                        tot += s_norm_partial[4 + w * L + lane_id]
                    s_rn[lane_id] = cute.math.rsqrt(tot + Float32(1e-06))
                if consumer_warp == 1 and lane_id == 0:
                    qt = (
                        s_norm_partial[0]
                        + s_norm_partial[1]
                        + s_norm_partial[2]
                        + s_norm_partial[3]
                    )
                    s_misc[6] = cute.math.rsqrt(qt + Float32(1e-06)) * scale
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                s_qhat[k_idx] = q_raw * s_misc[6]
                for j in cutlass.range_constexpr(L):
                    khj = kwin[j] * s_rn[j]
                    s_khat32[j * self.k_dim + k_idx] = khj
                    s_kq16[j, (k_idx % elems, k_idx // elems)] = Float16(khj)
                for unit in cutlass.range_constexpr(self.dot_k // 8):
                    frag = cute.make_fragment(8, Float16)
                    for jj in cutlass.range_constexpr(8):
                        frag[jj] = Float16(kwin[unit * 8 + jj] * s_rn[unit * 8 + jj])
                    cute.autovec_copy(frag, s_dk_u[tmem_row, ((None, unit), 1)])
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(k16_full)
                cute.arch.barrier(barrier_id=3, number_of_threads=128)
                e = tid - 64
                if e < 120:
                    ri = Int32(0)
                    rj = Int32(0)
                    acc_e = e
                    for r in cutlass.range_constexpr(L - 1):
                        rowlen = L - 1 - r
                        if acc_e >= 0 and acc_e < rowlen:
                            ri = Int32(r)
                            rj = Int32(r + 1) + acc_e
                        acc_e = acc_e - rowlen
                    gacc = Float32(0.0)
                    for kk in cutlass.range_constexpr(self.k_dim):
                        gacc += (
                            s_khat32[ri * self.k_dim + kk]
                            * s_khat32[rj * self.k_dim + kk]
                        )
                    s_gram[ri * L + rj] = gacc
                if consumer_warp == 0 and lane_id == 0:
                    par = Int32(sg_parity[state_idx, key_head])
                    am = Float32(0.0)
                    for gg in cutlass.range_constexpr(self.h):
                        a2 = Float32(sg_meta[state_idx, par, gg])
                        if a2 > am:
                            am = a2
                    s_g = am / Float32(E2M1_MAX * E4M3_MAX)
                    if s_g <= Float32(0.0):
                        s_g = Float32(1.0)
                    s_misc[1] = s_g
                    s_misc[2] = rcp_rn_f32(Float32(6.0) * s_g)
                    s_misc[3] = rcp_rn_f32(s_g)
                    s_misc[5] = Float32(par)
                    sg_meta[state_idx, 2, key_head] = s_g
                cute.arch.barrier(barrier_id=4, number_of_threads=128)
                if consumer_warp < gqa:
                    value_head_w = first_head + consumer_warp
                    c = lane_id % L
                    gw = g_cache[state_idx, value_head_w, c]
                    bw = beta_cache[state_idx, value_head_w, c]
                    if c >= wp:
                        gw = Float32(0.0)
                        bw = Float32(0.0)
                    if c == wp:
                        x = Float32(a[batch_id, value_head_w]) + Float32(
                            dt_bias[value_head_w]
                        )
                        sp = x
                        if x <= Float32(20.0):
                            sp = cute.math.log(
                                Float32(1.0) + cute.math.exp(x, fastmath=True),
                                fastmath=True,
                            )
                        gw = (
                            -cute.math.exp(Float32(a_log[value_head_w]), fastmath=True)
                            * sp
                        )
                        bw = Float32(
                            BFloat16(
                                Float32(1.0)
                                / (
                                    Float32(1.0)
                                    + cute.math.exp(
                                        -Float32(b[batch_id, value_head_w]),
                                        fastmath=True,
                                    )
                                )
                            )
                        )
                    prefix = gw
                    for offset in (1, 2, 4, 8):
                        lower = cute.arch.shuffle_sync_up(
                            prefix, offset, mask_and_clamp=0
                        )
                        if c >= offset:
                            prefix += lower
                    total = cute.arch.shuffle_sync(prefix, 15)
                    if lane_id < L:
                        s_G[consumer_warp * L + c] = prefix
                        eg = Float32(0.0)
                        dw = Float32(0.0)
                        if c <= wp:
                            eg = cute.math.exp(prefix, fastmath=True)
                            dw = cute.math.exp(total - prefix, fastmath=True)
                        s_expG[consumer_warp * L + c] = eg
                        s_dw[consumer_warp * L + c] = dw
                        s_bj[consumer_warp * L + c] = bw
                    if lane_id == 0:
                        s_lamh[consumer_warp] = cute.math.exp(total, fastmath=True)
                cute.arch.barrier(barrier_id=5, number_of_threads=128)
                for e2 in cutlass.range_constexpr(2 * gqa):
                    e = tid - 64 + e2 * 128
                    ph = e // (L * L)
                    kj = e % (L * L) // L
                    ki = e % L
                    cval = Float32(0.0)
                    if ki < kj and kj <= wp:
                        cval = (
                            cute.math.exp(
                                s_G[ph * L + kj] - s_G[ph * L + ki], fastmath=True
                            )
                            * s_gram[ki * L + kj]
                        )
                    s_coef[e] = cval
                cute.arch.barrier(barrier_id=6, number_of_threads=128)
                rcp6sg = s_misc[2]
                rcp_sg = s_misc[3]
                thread_amax = Float32(0.0)
                for p in cutlass.range_constexpr(gqa):
                    value_head = first_head + p
                    vrow = cute.make_rmem_tensor((L,), Float32)
                    for j in cutlass.range_constexpr(L):
                        vrow[j] = Float32(v_cache[state_idx, value_head, j, row])
                    vcur = Float32(v[batch_id, value_head, row])
                    lam = s_lamh[p]
                    cute.arch.mbarrier_wait(f_stepa[p], 0)
                    _tcgen05.fence_after_thread_sync()
                    tmem_base = tmem_addr[0]
                    kh = _tcgen05.ld(tmem_row, tmem_base, "32x32b", 16)
                    _tcgen05.wait_ld()
                    _tcgen05.fence_before_thread_sync()
                    uvals = cute.make_rmem_tensor((L,), Float32)
                    for j in cutlass.range_constexpr(L):
                        vj = vrow[j]
                        if j >= wp:
                            vj = Float32(0.0)
                        if j == wp:
                            vj = vcur
                        cres = vj - s_expG[p * L + j] * kh[j]
                        sacc = Float32(0.0)
                        for i in cutlass.range_constexpr(j):
                            sacc += s_coef[p * L * L + j * L + i] * uvals[i]
                        uvals[j] = s_bj[p * L + j] * (cres - sacc)
                    for unit in cutlass.range_constexpr(self.dot_k // 8):
                        frag = cute.make_fragment(8, Float16)
                        for jj in cutlass.range_constexpr(8):
                            frag[jj] = Float16(
                                uvals[unit * 8 + jj] * s_dw[p * L + unit * 8 + jj]
                            )
                        cute.autovec_copy(frag, s_dk_u[row, ((None, unit), 0)])
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(fd[p])
                    o_acc = Float32(0.0)
                    row_amax = Float32(0.0)
                    slo_row = s_lo.iterator.toint() + (
                        Int64(state_idx * self.hv + value_head) * Int64(self.v_dim)
                        + Int64(row)
                    ) * Int64(self.k_dim // 2)
                    sf_words = cute.make_rmem_tensor((2,), Uint32)
                    sf_words[0] = Uint32(0)
                    sf_words[1] = Uint32(0)
                    for g in cutlass.range_constexpr(2):
                        if cutlass.const_expr(g == 0):
                            cute.arch.mbarrier_wait(f_mma_a[p], 0)
                        else:
                            cute.arch.mbarrier_wait(f_mma_b[p], 0)
                        _tcgen05.fence_after_thread_sync()
                        for chunk in cutlass.range_constexpr(elems // 16):
                            update = _tcgen05.ld(
                                tmem_row, tmem_base + chunk * 16, "32x32b", 16
                            )
                            _tcgen05.wait_ld()
                            vals = cute.make_rmem_tensor((16,), Float32)
                            for u in cutlass.range_constexpr(16 // uw):
                                s_unit = s_state_u[
                                    row, ((None, chunk * (16 // uw) + u), g)
                                ]
                                old = cute.make_fragment(uw, Float16)
                                cute.autovec_copy(s_unit, old)
                                for jj in cutlass.range_constexpr(uw):
                                    nv = Float32(old[jj]) * lam + update[u * uw + jj]
                                    vals[u * uw + jj] = nv
                                    o_acc += (
                                        nv
                                        * s_qhat[g * elems + chunk * 16 + u * uw + jj]
                                    )
                                k_group8 = g * (elems // uw) + chunk * (16 // uw) + u
                                spatial = Uint32(
                                    (value_head * self.v_dim + row) * (self.k_dim // uw)
                                    + k_group8
                                )
                                r0, r1, r2, r3 = _philox4x32(
                                    Uint32(state_idx),
                                    spatial,
                                    event_lo,
                                    event_hi,
                                    sr_key0,
                                    sr_key1,
                                    7,
                                )
                                packed = cute.make_fragment(4, Uint32)
                                packed[0] = _cvt_f32x2_f16x2_rs(
                                    vals[u * uw], vals[u * uw + 1], r0
                                )
                                packed[1] = _cvt_f32x2_f16x2_rs(
                                    vals[u * uw + 2], vals[u * uw + 3], r1
                                )
                                packed[2] = _cvt_f32x2_f16x2_rs(
                                    vals[u * uw + 4], vals[u * uw + 5], r2
                                )
                                packed[3] = _cvt_f32x2_f16x2_rs(
                                    vals[u * uw + 6], vals[u * uw + 7], r3
                                )
                                cute.autovec_copy(
                                    cute.recast_tensor(packed, Float16), s_unit
                                )
                            bamax = cute.math.absf(vals[0])
                            for i in cutlass.range_constexpr(1, 16):
                                bamax = cute.arch.fmax(bamax, cute.math.absf(vals[i]))
                            row_amax = cute.arch.fmax(row_amax, bamax)
                            sbb = f32_to_e4m3_byte(bamax * rcp6sg)
                            sb_val = e4m3_byte_to_f32(sbb)
                            if sb_val <= Float32(0.0):
                                sb_val = Float32(1.0)
                            inv = rcp_rn_f32(sb_val) * rcp_sg
                            qw0 = f32x8_to_e2m1x8(
                                vals[0] * inv,
                                vals[1] * inv,
                                vals[2] * inv,
                                vals[3] * inv,
                                vals[4] * inv,
                                vals[5] * inv,
                                vals[6] * inv,
                                vals[7] * inv,
                            )
                            qw1 = f32x8_to_e2m1x8(
                                vals[8] * inv,
                                vals[9] * inv,
                                vals[10] * inv,
                                vals[11] * inv,
                                vals[12] * inv,
                                vals[13] * inv,
                                vals[14] * inv,
                                vals[15] * inv,
                            )
                            blk = g * 4 + chunk
                            st_global_v2_u32(slo_row + blk * 8, qw0, qw1)
                            sf_words[blk // 4] = sf_words[blk // 4] | sbb << Uint32(
                                blk % 4 * 8
                            )
                        _tcgen05.fence_before_thread_sync()
                        if cutlass.const_expr(g == 0):
                            cute.arch.mbarrier_arrive(f_epi_a[p])
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(f_gdone[p * 2 + g])
                    sf_row = (
                        sf_lo.iterator.toint()
                        + Int64(state_idx * self.hv + value_head) * Int64(1024)
                        + Int64(row % 32 * 16 + row // 32 % 4 * 4)
                    )
                    st_global_u32(sf_row, sf_words[0])
                    st_global_u32(sf_row + 512, sf_words[1])
                    out[batch_id, value_head, row] = BFloat16(o_acc)
                    if row_amax > thread_amax:
                        thread_amax = row_amax
                wmax = cute.arch.warp_reduction_max(thread_amax)
                if lane_id == 0:
                    s_gmax[consumer_warp] = wmax
                cute.arch.barrier(barrier_id=7, number_of_threads=128)
                if consumer_warp == 0 and lane_id == 0:
                    gm = s_gmax[0]
                    for w in cutlass.range_constexpr(1, 4):
                        if s_gmax[w] > gm:
                            gm = s_gmax[w]
                    par = Int32(s_misc[5])
                    sg_meta[state_idx, par ^ 1, key_head] = gm
                    sg_parity[state_idx, key_head] = par ^ 1

    @cache
    @staticmethod
    def compile(h, hv, k_dim=128, v_dim=128, cache_len=16):
        batch = cute.sym_int()
        slots = cute.sym_int()
        q = make_fake_tensor(BFloat16, (batch, h, k_dim), divisibility=16)
        k = make_fake_tensor(BFloat16, (batch, h, k_dim), divisibility=16)
        v = make_fake_tensor(BFloat16, (batch, hv, v_dim), divisibility=16)
        a = make_fake_tensor(BFloat16, (batch, hv), divisibility=16)
        b = make_fake_tensor(BFloat16, (batch, hv), divisibility=16)
        a_log = make_fake_tensor(Float32, (hv,), divisibility=16)
        dt_bias = make_fake_tensor(Float32, (hv,), divisibility=16)
        state = make_fake_tensor(Float16, (slots, hv, v_dim, k_dim), divisibility=16)
        v_cache = make_fake_tensor(
            BFloat16, (slots, hv, cache_len, v_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            BFloat16, (slots, h, cache_len, k_dim), divisibility=16
        )
        g_cache = make_fake_tensor(Float32, (slots, hv, cache_len), divisibility=16)
        beta_cache = make_fake_tensor(Float32, (slots, hv, cache_len), divisibility=16)
        s_lo = make_fake_tensor(Int8, (slots, hv, v_dim, k_dim // 2), divisibility=16)
        sf_lo = make_fake_tensor(Int8, (slots, hv, 1024), divisibility=16)
        sg_meta = make_fake_tensor(Float32, (slots, 3, h), divisibility=1)
        sg_parity = make_fake_tensor(Int32, (slots, h), divisibility=1)
        state_indices = make_fake_tensor(Int32, (batch,), divisibility=1)
        write_pos = make_fake_tensor(Int64, (batch,), divisibility=1)
        is_flush = make_fake_tensor(Int8, (batch,), divisibility=1)
        out = make_fake_tensor(BFloat16, (batch, hv, v_dim), divisibility=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm103GdnTp4FlushKernel(h, hv, k_dim, v_dim, cache_len),
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            v_cache,
            k_cache,
            g_cache,
            beta_cache,
            s_lo,
            sf_lo,
            sg_meta,
            sg_parity,
            state_indices,
            write_pos,
            is_flush,
            out,
            Float32(1.0),
            Uint32(0),
            Uint32(0),
            stream,
            options="--enable-tvm-ffi",
        )


def gdn_nvfp4_flush(
    q,
    k,
    v,
    a,
    b,
    a_log,
    dt_bias,
    scale,
    state,
    v_cache,
    k_cache,
    g_cache,
    beta_cache,
    state_lo,
    sf_lo,
    sg_meta,
    sg_parity,
    state_indices,
    write_pos,
    is_flush,
    out,
    *,
    stochastic_rounding_seed=0,
    stochastic_rounding_salt=0,
):
    _, h, k_dim = q.shape
    _, hv, v_dim = v.shape
    cache_len = v_cache.shape[2]
    if state.dtype != torch.float16:
        raise TypeError("the anchor must be FP16")
    if write_pos.dtype != torch.int64:
        raise TypeError("stochastic rounding requires int64 packed position tags")
    Sm103GdnTp4FlushKernel.compile(h, hv, k_dim, v_dim, cache_len)(
        q,
        k,
        v,
        a,
        b,
        a_log.float(),
        dt_bias.float(),
        state,
        v_cache,
        k_cache,
        g_cache,
        beta_cache,
        state_lo.view(torch.int8),
        sf_lo.view(torch.int8),
        sg_meta,
        sg_parity,
        state_indices,
        write_pos,
        is_flush.view(torch.int8),
        out,
        float(scale),
        stochastic_rounding_seed & 4294967295,
        stochastic_rounding_salt & 4294967295,
    )
    return out
