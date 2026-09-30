# SPDX-License-Identifier: Apache-2.0
"""Mamba2 FP32 non-flush comparator; L=16."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int8, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor

from low_precision.common import _tcgen05, simple_tma_copy
from low_precision.common.dsl import alloc_tmem as _alloc_tmem
from low_precision.common.dsl import dealloc_tmem as _dealloc_tmem
from low_precision.common.dsl import make_rmem
from low_precision.common.dsl import make_tf32_idesc as _make_tf32_idesc
from low_precision.common.dsl import mma_tf32 as _mma_tf32
from low_precision.common.sync import _exit_if_nonzero


class Sm103Mamba2ReplayFusedKernel:
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
        self.ngroups = ngroups
        self.hpg = hpg
        self.pairs = hpg // 2
        self.head_dim = head_dim
        self.dstate = dstate
        self.cache_len = cache_len
        self.num_warps = 6
        self.mma_n = 16
        self.elems = 32
        self.n_span = dstate // self.elems
        self.n_slots = self.pairs * self.n_span

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
        stream: CUstream,
    ):
        state_tma = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        self.kernel(
            state_tma,
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
    ):
        state_t = Float32
        elems = self.elems
        n_span = self.n_span
        n_slots = self.n_slots
        pairs = self.pairs
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
        cop_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (self.mma_n, (elems, n_span)), stride=(elems, (1, self.mma_n * elems))
            ),
        )
        s_cop = allocate_tensor(smem, state_t, cop_layout)
        s_c = smem.allocate_array(Float32, self.dstate)
        s_bc = smem.allocate_array(Float32, L)
        s_scale = smem.allocate_array(Float32, self.hpg * L)
        s_tdecay = smem.allocate_array(Float32, self.hpg)
        qk_full = smem.allocate_array(Int64, 1)
        state_slot = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        mma_slot = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        epilogue_done = smem.allocate_array(Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)

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
        ):
            hh = p * 2 + head_par
            h_glob = first_head + hh
            x_raw = x[batch_id, h_glob, r]
            x_cur = Float32(x_raw)
            x_cache[state_idx, h_glob, wp, r] = x_raw
            acc = x_cur * Float32(D[h_glob, r])
            for c in cutlass.range_constexpr(L_):
                if c <= wp:
                    xc = Float32(x_cache[state_idx, h_glob, c, r])
                    acc += xc * (s_scale[hh * L_ + c] * s_bc[c])
            accs[p % 2] = acc

        state_idx = state_indices[batch_id]
        wp = write_pos[batch_id]
        _exit_if_nonzero(Int32(is_flush[batch_id]))
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(state_slot[0], 1)
                cute.arch.mbarrier_init(state_slot[1], 1)
                cute.arch.mbarrier_init_fence()
            cpasync.prefetch_descriptor(state_tma.atom)
            for slot in cutlass.range_constexpr(2):
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
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(qk_full, 128)
                for slot in cutlass.range_constexpr(2, n_slots):
                    cute.arch.mbarrier_init(state_slot[slot], 1)
                for slot in cutlass.range_constexpr(n_slots):
                    cute.arch.mbarrier_init(mma_slot[slot], 1)
                cute.arch.mbarrier_init(epilogue_done, 128 * pairs)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp_id == 0:
            for slot in cutlass.range_constexpr(2, n_slots):
                p = slot // n_span
                g = slot % n_span
                cute.arch.mbarrier_wait(mma_slot[slot - 2], 0)
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
        elif warp_id == 1:
            _alloc_tmem(tmem_addr, 128)
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_base = tmem_addr[0]
            cute.arch.mbarrier_wait(qk_full, 0)
            sdesc = _tcgen05.make_sdesc_128B_swizzle(2 * self.head_dim * 128)
            cop_base = sdesc | s_cop.iterator.toint() >> 4
            state_base0 = sdesc | s_state.iterator.toint() >> 4
            state_base1 = sdesc | s_state.iterator.toint() + 16384 >> 4
            idesc = _make_tf32_idesc(2 * self.head_dim, self.mma_n)
            for p in cutlass.range_constexpr(pairs):
                for g in cutlass.range_constexpr(n_span):
                    slot = p * n_span + g
                    cute.arch.mbarrier_wait(state_slot[slot], 0)
                    _tcgen05.fence_after_thread_sync()
                    stage_base = state_base0 if slot % 2 == 0 else state_base1
                    for k_atom in cutlass.range_constexpr(4):
                        state_desc = stage_base | k_atom * 32 >> 4
                        cop_desc = cop_base | g * self.mma_n * 128 + k_atom * 32 >> 4
                        d_col = tmem_base + p * self.mma_n
                        _mma_tf32(
                            d_col, state_desc, cop_desc, idesc, g > 0 or k_atom > 0
                        )
                    _tcgen05.commit(mma_slot[slot])
            cute.arch.mbarrier_wait(epilogue_done, 0)
            _dealloc_tmem(tmem_addr[0], 128)
        else:
            consumer_warp = warp_id - 2
            tmem_row = consumer_warp * 32 + lane_id
            row = tmem_row ^ 64
            head_par = row // self.head_dim
            r = row % self.head_dim
            n_idx = tmem_row
            c_val = Float32(C[batch_id, group, n_idx])
            s_cop[0, (n_idx % elems, n_idx // elems)] = state_t(c_val)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(qk_full)
            s_c[n_idx] = c_val
            b_val = B[batch_id, group, n_idx]
            B_cache[state_idx, group, wp, n_idx] = b_val
            if consumer_warp == 0 and lane_id < self.hpg:
                h_glob = first_head + lane_id
                dtv = Float32(dt[batch_id, h_glob]) + Float32(dt_bias[h_glob])
                if dtv <= Float32(20.0):
                    dtv = cute.math.log(
                        Float32(1.0) + cute.math.exp(dtv, fastmath=True), fastmath=True
                    )
                dt_cache[state_idx, h_glob, wp] = dtv
            cute.arch.barrier(barrier_id=6, number_of_threads=128)
            for cg in cutlass.range_constexpr(4):
                c_idx = consumer_warp + cg * 4
                acc = Float32(0.0)
                if c_idx <= wp:
                    for kg in cutlass.range_constexpr(4):
                        nn = lane_id + kg * 32
                        acc += Float32(B_cache[state_idx, group, c_idx, nn]) * s_c[nn]
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
                    )
                cute.arch.mbarrier_wait(mma_slot[p * n_span + n_span - 1], 0)
                _tcgen05.fence_after_thread_sync()
                tmem_base = tmem_addr[0]
                r_col = tmem_base + p * self.mma_n
                readout = _tcgen05.ld(tmem_row, r_col, "32x32b", 1)
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
            Float32, (slots, nheads, head_dim, dstate), divisibility=16
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
        write_pos = make_fake_tensor(Int32, (batch,), divisibility=1)
        is_flush = make_fake_tensor(Int8, (batch,), divisibility=1)
        out = make_fake_tensor(BFloat16, (batch, nheads, head_dim), divisibility=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm103Mamba2ReplayFusedKernel(nheads, ngroups, head_dim, dstate, cache_len),
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
            stream,
            options="--enable-tvm-ffi",
        )


def mamba2_fp32(
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
) -> torch.Tensor:
    slots, H, P, N = state.shape
    G = B.shape[1]
    L = x_cache.shape[2]
    if state.dtype != torch.float32:
        raise TypeError("state must be torch.float32")
    if is_flush.dtype == torch.bool:
        is_flush = is_flush.view(torch.int8)
    Sm103Mamba2ReplayFusedKernel.compile(H, G, P, N, L)(
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
    )
    return out
