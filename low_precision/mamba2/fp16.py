# SPDX-License-Identifier: Apache-2.0
"""MAMBA2 fp16; fixed L=16 specialization."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float16, Float32, Int8, Int32, Int64, Uint32, Uint64, cute
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor

from low_precision.common import _tcgen05, simple_tma_copy
from low_precision.common.dsl import (
    alloc_tmem as _alloc_tmem,
)
from low_precision.common.dsl import (
    dealloc_tmem as _dealloc_tmem,
)
from low_precision.common.dsl import (
    make_fp16_idesc as _make_fp16_idesc,
)
from low_precision.common.dsl import (
    make_rmem,
)
from low_precision.common.dsl import (
    make_sdesc_32B_swizzle as _make_sdesc_32B_swizzle,
)
from low_precision.common.philox import (
    cvt_f32x2_f16x2_rs,
    philox4x32,
)


class Sm103Mamba2ReplayFp16Kernel:
    def __init__(
        self,
        nheads: int,
        ngroups: int,
        head_dim: int = 64,
        dstate: int = 128,
        cache_len: int = 16,
    ) -> None:
        assert head_dim == 64 and dstate == 128 and (cache_len == 16)
        assert nheads % ngroups == 0
        hpg = nheads // ngroups
        assert hpg == 16, "kernel is specialized for 16 heads per group"
        self.nheads = nheads
        self.ngroups = ngroups
        self.hpg = hpg
        self.pairs = hpg // 2
        self.head_dim = head_dim
        self.dstate = dstate
        self.cache_len = cache_len
        self.dot_k = 16
        self.num_warps = 6
        self.mma_n = 16
        self.elems = 64
        self.n_span = dstate // self.elems
        self.n_slots = self.pairs * self.n_span
        self.unit = 8
        self.tmem_cols = 128
        self.n_acc = self.tmem_cols // self.elems

    @cute.jit
    def _make_state_tma(self, state: cute.Tensor, op: cpasync.TmaCopyOp):
        num_elems = 128 // (state.element_type.width // 8)
        swizzle = cute.make_swizzle(3, 4, 3)
        smem_layout = cute.make_layout(
            (1, 1, self.head_dim, (num_elems, 1), 1),
            stride=(
                0,
                0,
                num_elems,
                (1, self.head_dim * num_elems),
                self.head_dim * num_elems,
            ),
        )
        smem_layout = cute.make_composed_layout(swizzle, 0, smem_layout)
        return cpasync.make_tiled_tma_atom(
            op,
            cute.logical_divide(state, (None, None, None, num_elems)),
            smem_layout,
            cta_tiler=(1, 1, self.head_dim, num_elems),
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        dt: cute.Tensor,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        D: cute.Tensor,
        dt_bias: cute.Tensor,
        state: cute.Tensor,
        x_cache: cute.Tensor,
        dt_cache: cute.Tensor,
        B_cache: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        sr_key0: Uint32,
        sr_key1: Uint32,
        stream: CUstream,
    ):
        state_tma = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        state_tma_store = self._make_state_tma(state, cpasync.CopyBulkTensorTileS2GOp())
        self.kernel(
            state_tma,
            state_tma_store,
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            x_cache,
            dt_cache,
            B_cache,
            state_indices,
            write_pos,
            is_flush,
            out,
            sr_key0,
            sr_key1,
        ).launch(
            grid=(x.shape[0], self.ngroups, 1),
            block=(self.num_warps * 32, 1, 1),
            min_blocks_per_mp=4,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state_tma: cpasync.TmaInfo,
        state_tma_store: cpasync.TmaInfo,
        x: cute.Tensor,
        dt: cute.Tensor,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        D: cute.Tensor,
        dt_bias: cute.Tensor,
        x_cache: cute.Tensor,
        dt_cache: cute.Tensor,
        B_cache: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        sr_key0: Uint32,
        sr_key1: Uint32,
    ):
        state_t = Float16
        elems = self.elems
        n_span = self.n_span
        n_slots = self.n_slots
        n_acc = self.n_acc
        pairs = self.pairs
        uw = self.unit
        L = self.cache_len
        tid, _, _ = cute.arch.thread_idx()
        batch_id, group, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        first_head = group * self.hpg

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        state_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                ((self.head_dim, 2), (elems, 2)),
                stride=((elems, self.head_dim * elems), (1, 2 * self.head_dim * elems)),
            ),
        )
        s_state = allocate_tensor(smem, state_t, state_layout)
        state_unit_layout = cute.make_layout(
            (2 * self.head_dim, ((uw, elems // uw), 2)),
            stride=(elems, ((1, uw), 2 * self.head_dim * elems)),
        )
        s_state_u = cute.make_tensor(s_state.iterator, state_unit_layout)
        cop_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (self.mma_n, (elems, n_span)), stride=(elems, (1, self.mma_n * elems))
            ),
        )
        s_cop = allocate_tensor(smem, state_t, cop_layout)
        rows = 2 * self.head_dim
        dot_layout = cute.make_layout((rows, self.dot_k), stride=(self.dot_k, 1))
        dot_unit_layout = cute.make_layout(
            (rows, (8, self.dot_k // 8)), stride=(self.dot_k, (1, 8))
        )
        s_xall1 = smem.allocate_tensor(
            BFloat16, cute.make_layout(2048), byte_alignment=1024
        )
        s_ball_extra = smem.allocate_tensor(
            BFloat16, cute.make_layout(2048), byte_alignment=1024
        )
        xall_bases = [
            cute.recast_ptr(
                cute.recast_ptr(s_cop.iterator, dtype=BFloat16),
                cute.make_swizzle(1, 4, 3),
                dtype=BFloat16,
            ),
            cute.recast_ptr(
                s_xall1.iterator, cute.make_swizzle(1, 4, 3), dtype=BFloat16
            ),
        ]
        ball_base = cute.recast_ptr(
            s_ball_extra.iterator, cute.make_swizzle(1, 4, 3), dtype=BFloat16
        )
        s_xall = [cute.make_tensor(b, dot_layout) for b in xall_bases]
        s_ball = cute.make_tensor(ball_base, dot_layout)
        s_ball_u = cute.make_tensor(ball_base, dot_unit_layout)
        s_c = smem.allocate_array(Float32, self.dstate)
        s_bc = smem.allocate_array(Float32, L)
        s_scale = smem.allocate_array(Float32, self.hpg * L)
        s_tdecay = smem.allocate_array(Float32, self.hpg)
        qk_full = smem.allocate_array(Int64, 1)
        state_slot = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        mma_slot = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        epilogue_done = smem.allocate_array(Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)
        f_mma = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        f_gdone = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        fd = [smem.allocate_array(Int64, 1) for _ in range(pairs)]
        fk_full = smem.allocate_array(Int64, 1)

        def state_load(
            slot, n_span, first_head, state_idx, state_tma, s_state, state_slot
        ):
            p = slot // n_span
            g = slot % n_span
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(state_slot[slot], 16384)
            for hp in cutlass.range_constexpr(2):
                head = first_head + p * 2 + hp
                g_state = state_tma.tma_tensor[state_idx, head, None, (None, g)]
                simple_tma_copy(
                    state_tma.atom,
                    g_state,
                    s_state[(None, hp), (None, slot % 2)],
                    state_slot[slot],
                )

        def stage_x(
            p, cidx, wp, state_idx, first_head, s_scale, x_cache, s_xall, fd, L_
        ):
            hp = cidx // 64
            lg = cidx % 64 // 8
            r0 = cidx % 64 % 8 * 8
            st_hh = p * 2 + hp
            st_h = first_head + st_hh
            xrow0 = cute.logical_divide(
                x_cache[state_idx, st_h, lg, None], cute.make_layout(8)
            )
            xrow1 = cute.logical_divide(
                x_cache[state_idx, st_h, lg + 8, None], cute.make_layout(8)
            )
            f0 = make_rmem(8, BFloat16)
            f1 = make_rmem(8, BFloat16)
            cute.autovec_copy(xrow0[None, r0 // 8], f0)
            cute.autovec_copy(xrow1[None, r0 // 8], f1)
            s0 = Float32(0.0)
            if lg <= wp:
                s0 = s_scale[st_hh * L_ + lg]
            s1 = Float32(0.0)
            if lg + 8 <= wp:
                s1 = s_scale[st_hh * L_ + lg + 8]
            for j in cutlass.range_constexpr(8):
                row_j = hp * 64 + r0 + j
                s_xall[p % 2][row_j, lg] = BFloat16(Float32(f0[j]) * s0)
                s_xall[p % 2][row_j, lg + 8] = BFloat16(Float32(f1[j]) * s1)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(fd[p])

        def pair_walk(
            p,
            accs,
            batch_id,
            head_par,
            r,
            wp,
            state_idx,
            first_head,
            s_scale,
            s_bc,
            x,
            x_cache,
            D,
            L_,
            live_,
        ):
            hh = p * 2 + head_par
            h_glob = first_head + hh
            x_raw = x[batch_id, h_glob, r]
            if live_:
                x_cache[state_idx, h_glob, wp, r] = x_raw
            acc = Float32(x_raw) * Float32(D[h_glob, r])
            for c in cutlass.range_constexpr(L_):
                xc = Float32(x_cache[state_idx, h_glob, c, r])
                acc += xc * (s_scale[hh * L_ + c] * s_bc[c])
            accs[p % 2] = acc

        state_idx = state_indices[batch_id]
        tag = Uint64(write_pos[batch_id])
        wp = Int32(tag & Uint64(self.cache_len - 1))
        sr_event_lo = Uint32(tag >> Uint64(4))
        sr_event_hi = Uint32(tag >> Uint64(36))
        flush = is_flush[batch_id] != 0
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(state_slot[0], 1)
                cute.arch.mbarrier_init(state_slot[1], 1)
                cute.arch.mbarrier_init_fence()
            cpasync.prefetch_descriptor(state_tma.atom)
            for slot in cutlass.range_constexpr(2):
                state_load(
                    slot, n_span, first_head, state_idx, state_tma, s_state, state_slot
                )
            with cute.arch.elect_one():
                for slot in cutlass.range_constexpr(2, n_slots):
                    cute.arch.mbarrier_init(state_slot[slot], 1)
                if flush:
                    for slot in cutlass.range_constexpr(n_slots):
                        cute.arch.mbarrier_init(f_mma[slot], 1)
                        cute.arch.mbarrier_init(f_gdone[slot], 128)
                    cute.arch.mbarrier_init(fk_full, 128)
                    for i in cutlass.range_constexpr(pairs):
                        cute.arch.mbarrier_init(fd[i], 128)
                else:
                    cute.arch.mbarrier_init(qk_full, 128)
                    for slot in cutlass.range_constexpr(n_slots):
                        cute.arch.mbarrier_init(mma_slot[slot], 1)
                    cute.arch.mbarrier_init(epilogue_done, 128 * pairs)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp_id == 0:
            if flush:
                cpasync.prefetch_descriptor(state_tma_store.atom)
                for slot in cutlass.range_constexpr(n_slots):
                    p = slot // n_span
                    g = slot % n_span
                    cute.arch.mbarrier_wait(f_gdone[slot], 0)
                    for hp in cutlass.range_constexpr(2):
                        head = first_head + p * 2 + hp
                        g_out = state_tma_store.tma_tensor[
                            state_idx, head, None, (None, g)
                        ]
                        simple_tma_copy(
                            state_tma_store.atom,
                            s_state[(None, hp), (None, slot % 2)],
                            g_out,
                        )
                    cute.arch.cp_async_bulk_commit_group()
                    if cutlass.const_expr(slot + 2 < n_slots):
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        state_load(
                            slot + 2,
                            n_span,
                            first_head,
                            state_idx,
                            state_tma,
                            s_state,
                            state_slot,
                        )
                cute.arch.cp_async_bulk_wait_group(0)
            else:
                for slot in cutlass.range_constexpr(2, n_slots):
                    cute.arch.mbarrier_wait(mma_slot[slot - 2], 0)
                    state_load(
                        slot,
                        n_span,
                        first_head,
                        state_idx,
                        state_tma,
                        s_state,
                        state_slot,
                    )
        elif warp_id == 1:
            _alloc_tmem(tmem_addr, self.tmem_cols)
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_base = tmem_addr[0]
            if flush:
                cute.arch.mbarrier_wait(fk_full, 0)
                d_bases = [
                    _make_sdesc_32B_swizzle(Int64(s.iterator.toint() >> 4))
                    for s in s_xall
                ]
                b_base = _make_sdesc_32B_swizzle(Int64(s_ball.iterator.toint() >> 4))
                idesc_up = _tcgen05.make_bf16_idesc(2 * self.head_dim, elems)
                for p in cutlass.range_constexpr(pairs):
                    cute.arch.mbarrier_wait(fd[p], 0)
                    for g in cutlass.range_constexpr(n_span):
                        slot = p * n_span + g
                        if cutlass.const_expr(slot >= n_acc):
                            cute.arch.mbarrier_wait(f_gdone[slot - n_acc], 0)
                        _tcgen05.fence_after_thread_sync()
                        _tcgen05.mma_f16(
                            tmem_base + slot % n_acc * elems,
                            d_bases[p % 2],
                            b_base + (g * elems * 2 * self.dot_k >> 4),
                            idesc_up,
                            False,
                        )
                        _tcgen05.commit(f_mma[slot])
                cute.arch.mbarrier_wait(f_gdone[n_slots - 1], 0)
            else:
                cute.arch.mbarrier_wait(qk_full, 0)
                sdesc = _tcgen05.make_sdesc_128B_swizzle(2 * self.head_dim * 128)
                cop_base = sdesc | s_cop.iterator.toint() >> 4
                state_base0 = sdesc | s_state.iterator.toint() >> 4
                state_base1 = sdesc | s_state.iterator.toint() + 16384 >> 4
                idesc = _make_fp16_idesc(2 * self.head_dim, self.mma_n)
                for p in cutlass.range_constexpr(pairs):
                    for g in cutlass.range_constexpr(n_span):
                        slot = p * n_span + g
                        cute.arch.mbarrier_wait(state_slot[slot], 0)
                        _tcgen05.fence_after_thread_sync()
                        stage_base = state_base0 if slot % 2 == 0 else state_base1
                        for k_atom in cutlass.range_constexpr(4):
                            state_desc = stage_base | k_atom * 32 >> 4
                            cop_desc = (
                                cop_base | g * self.mma_n * 128 + k_atom * 32 >> 4
                            )
                            _tcgen05.mma_f16(
                                tmem_base + p * self.mma_n,
                                state_desc,
                                cop_desc,
                                idesc,
                                g > 0 or k_atom > 0,
                            )
                        _tcgen05.commit(mma_slot[slot])
                cute.arch.mbarrier_wait(epilogue_done, 0)
            _dealloc_tmem(tmem_addr[0], self.tmem_cols)
        else:
            consumer_warp = warp_id - 2
            tmem_row = consumer_warp * 32 + lane_id
            row = tmem_row ^ 64
            head_par = row // self.head_dim
            r = row % self.head_dim
            n_idx = tmem_row
            c_val = Float32(C[batch_id, group, n_idx])
            if not flush:
                s_cop[0, (n_idx % elems, n_idx // elems)] = state_t(c_val)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(qk_full)
            s_c[n_idx] = c_val
            live = state_idx != 0
            b_val = B[batch_id, group, n_idx]
            if live:
                B_cache[state_idx, group, wp, n_idx] = b_val
            if consumer_warp == 0 and lane_id < self.hpg:
                h_glob = first_head + lane_id
                dtv = Float32(dt[batch_id, h_glob]) + Float32(dt_bias[h_glob])
                if dtv <= Float32(20.0):
                    dtv = cute.math.log(
                        Float32(1.0) + cute.math.exp(dtv, fastmath=True), fastmath=True
                    )
                if live:
                    dt_cache[state_idx, h_glob, wp] = dtv
            if flush:
                for p in cutlass.range_constexpr(pairs):
                    h_glob = first_head + p * 2 + head_par
                    if live:
                        x_cache[state_idx, h_glob, wp, r] = x[batch_id, h_glob, r]
            cute.arch.barrier(barrier_id=6, number_of_threads=128)
            if not flush:
                for cg in cutlass.range_constexpr(4):
                    c_idx = consumer_warp + cg * 4
                    acc = Float32(0.0)
                    if c_idx <= wp:
                        for kg in cutlass.range_constexpr(4):
                            nn = lane_id + kg * 32
                            acc += (
                                Float32(B_cache[state_idx, group, c_idx, nn]) * s_c[nn]
                            )
                    acc = cute.arch.warp_reduction_sum(acc)
                    if lane_id == 0:
                        s_bc[c_idx] = acc
                cute.arch.barrier(barrier_id=3, number_of_threads=128)
            for rnd in cutlass.range_constexpr(2):
                hh = rnd * 8 + consumer_warp * 2 + lane_id // 16
                seg = lane_id % 16
                h_glob = first_head + hh
                dtv = Float32(0.0)
                if seg <= wp:
                    dtv = dt_cache[state_idx, h_glob, seg]
                prefix = dtv
                for off in (1, 2, 4, 8):
                    lower = cute.arch.shuffle_sync_up(prefix, off, mask_and_clamp=0)
                    if seg >= off:
                        prefix += lower
                t15 = cute.arch.shuffle_sync(prefix, 15)
                t31 = cute.arch.shuffle_sync(prefix, 31)
                total = t15
                if lane_id >= 16:
                    total = t31
                A_h = Float32(A[h_glob])
                dA_total = A_h * total
                sc = dtv * cute.math.exp(dA_total - A_h * prefix, fastmath=True)
                if seg > wp:
                    sc = Float32(0.0)
                s_scale[hh * L + seg] = sc
                if seg == 0:
                    s_tdecay[hh] = cute.math.exp(dA_total, fastmath=True)
            cute.arch.barrier(barrier_id=4, number_of_threads=128)
            if flush:
                for unit in cutlass.range_constexpr(self.dot_k // 8):
                    frag = make_rmem(8, BFloat16)
                    for j in cutlass.range_constexpr(8):
                        c_idx = unit * 8 + j
                        bv = BFloat16(0.0)
                        if c_idx <= wp:
                            bv = B_cache[state_idx, group, c_idx, tmem_row]
                        frag[j] = bv
                    cute.autovec_copy(frag, s_ball_u[tmem_row, (None, unit)])
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(fk_full)
                stage_x(
                    0,
                    tmem_row,
                    wp,
                    state_idx,
                    first_head,
                    s_scale,
                    x_cache,
                    s_xall,
                    fd,
                    L,
                )
                for p in cutlass.range_constexpr(pairs):
                    if cutlass.const_expr(p + 1 < pairs):
                        stage_x(
                            p + 1,
                            tmem_row,
                            wp,
                            state_idx,
                            first_head,
                            s_scale,
                            x_cache,
                            s_xall,
                            fd,
                            L,
                        )
                    rmw_hh = p * 2 + head_par
                    h_glob = first_head + rmw_hh
                    total_decay = s_tdecay[rmw_hh]
                    oacc = Float32(x[batch_id, h_glob, r]) * Float32(D[h_glob, r])
                    for g in cutlass.range_constexpr(n_span):
                        slot = p * n_span + g
                        cute.arch.mbarrier_wait(f_mma[slot], 0)
                        cute.arch.mbarrier_wait(state_slot[slot], 0)
                        _tcgen05.fence_after_thread_sync()
                        acc_col = tmem_addr[0] + slot % n_acc * elems
                        for chunk in cutlass.range(elems // 16, unroll=1):
                            update = _tcgen05.ld(
                                tmem_row, acc_col + chunk * 16, "32x32b", 16
                            )
                            _tcgen05.wait_ld()
                            for u in cutlass.range_constexpr(16 // uw):
                                unit_idx = chunk * (16 // uw) + u
                                col0 = g * elems + unit_idx * uw
                                s_unit = s_state_u[row, ((None, unit_idx), slot % 2)]
                                old = make_rmem(uw, state_t)
                                cute.autovec_copy(s_unit, old)
                                spatial = Uint32(
                                    (h_glob * self.head_dim + r) * (self.dstate // uw)
                                    + col0 // uw
                                )
                                w = philox4x32(
                                    Uint32(state_idx),
                                    spatial,
                                    sr_event_lo,
                                    sr_event_hi,
                                    sr_key0,
                                    sr_key1,
                                    7,
                                )
                                packed = make_rmem(uw // 2, Uint32)
                                for pr in cutlass.range_constexpr(uw // 2):
                                    j = pr * 2
                                    lo = (
                                        Float32(old[j]) * total_decay
                                        + update[u * uw + j]
                                    )
                                    hi = (
                                        Float32(old[j + 1]) * total_decay
                                        + update[u * uw + j + 1]
                                    )
                                    oacc += lo * s_c[col0 + j]
                                    oacc += hi * s_c[col0 + j + 1]
                                    packed[pr] = cvt_f32x2_f16x2_rs(lo, hi, w[pr])
                                cute.autovec_copy(
                                    cute.recast_tensor(packed, Float16), s_unit
                                )
                        _tcgen05.fence_before_thread_sync()
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(f_gdone[slot])
                    out[batch_id, h_glob, r] = BFloat16(oacc)
            else:
                accs = make_rmem(2, Float32)
                pair_walk(
                    0,
                    accs,
                    batch_id,
                    head_par,
                    r,
                    wp,
                    state_idx,
                    first_head,
                    s_scale,
                    s_bc,
                    x,
                    x_cache,
                    D,
                    L,
                    live,
                )
                for p in cutlass.range_constexpr(pairs):
                    hh = p * 2 + head_par
                    h_glob = first_head + hh
                    if cutlass.const_expr(p + 1 < pairs):
                        pair_walk(
                            p + 1,
                            accs,
                            batch_id,
                            head_par,
                            r,
                            wp,
                            state_idx,
                            first_head,
                            s_scale,
                            s_bc,
                            x,
                            x_cache,
                            D,
                            L,
                            live,
                        )
                    cute.arch.mbarrier_wait(mma_slot[p * n_span + n_span - 1], 0)
                    _tcgen05.fence_after_thread_sync()
                    readout = _tcgen05.ld(
                        tmem_row, tmem_addr[0] + p * self.mma_n, "32x32b", 1
                    )
                    _tcgen05.wait_ld()
                    o = readout * s_tdecay[hh] + accs[p % 2]
                    out[batch_id, h_glob, r] = BFloat16(o)
                    _tcgen05.fence_before_thread_sync()
                    cute.arch.mbarrier_arrive(epilogue_done)

    @cache
    @staticmethod
    def compile(
        nheads: int,
        ngroups: int,
        head_dim: int = 64,
        dstate: int = 128,
        cache_len: int = 16,
    ):
        batch = cute.sym_int()
        slots = cute.sym_int()
        x = make_fake_tensor(BFloat16, (batch, nheads, head_dim), divisibility=16)
        dt = make_fake_tensor(Float32, (batch, nheads), divisibility=16)
        A = make_fake_tensor(Float32, (nheads,), divisibility=16)
        B = make_fake_tensor(BFloat16, (batch, ngroups, dstate), divisibility=16)
        C = make_fake_tensor(BFloat16, (batch, ngroups, dstate), divisibility=16)
        D = make_fake_tensor(Float32, (nheads, head_dim), divisibility=16)
        dt_bias = make_fake_tensor(Float32, (nheads,), divisibility=16)
        state = make_fake_tensor(
            Float16, (slots, nheads, head_dim, dstate), divisibility=16
        )
        x_cache = make_fake_tensor(
            BFloat16, (slots, nheads, cache_len, head_dim), divisibility=16
        )
        dt_cache = make_fake_tensor(
            Float32, (slots, nheads, cache_len), divisibility=16
        )
        B_cache = make_fake_tensor(
            BFloat16, (slots, ngroups, cache_len, dstate), divisibility=16
        )
        state_indices = make_fake_tensor(Int32, (batch,), divisibility=1)
        write_pos = make_fake_tensor(Int64, (batch,), divisibility=1)
        is_flush = make_fake_tensor(Int8, (batch,), divisibility=1)
        out = make_fake_tensor(BFloat16, (batch, nheads, head_dim), divisibility=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm103Mamba2ReplayFp16Kernel(nheads, ngroups, head_dim, dstate, cache_len),
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            state,
            x_cache,
            dt_cache,
            B_cache,
            state_indices,
            write_pos,
            is_flush,
            out,
            Uint32(0),
            Uint32(0),
            stream,
            options="--enable-tvm-ffi",
        )


def mamba2_fp16(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    out: torch.Tensor,
    stochastic_rounding_seed: int = 0,
    stochastic_rounding_salt: int = 0,
) -> torch.Tensor:
    slots, H, P, N = state.shape
    G = B.shape[1]
    L = x_cache.shape[2]
    if state.dtype is not torch.float16:
        raise TypeError("the state must be FP16")
    if write_pos.dtype != torch.int64:
        raise TypeError("stochastic rounding requires int64 packed position tags")
    if is_flush.dtype == torch.bool:
        is_flush = is_flush.view(torch.int8)
    Sm103Mamba2ReplayFp16Kernel.compile(H, G, P, N, L)(
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        state,
        x_cache,
        dt_cache,
        B_cache,
        state_indices,
        write_pos,
        is_flush,
        out,
        stochastic_rounding_seed & 4294967295,
        stochastic_rounding_salt & 4294967295,
    )
    return out
