# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, Vijay Thakkar, Tri Dao
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN fp32; fixed L=16 specialization."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor

from low_precision.common import _tcgen05, simple_tma_copy
from low_precision.common.gdn_ops import (
    _alloc_tmem,
    _dealloc_tmem,
    _make_tf32_idesc,
    _mma_tf32,
    _zero_output_and_exit_if_invalid,
)


class Sm103Fp16ReplayFusedKernel:
    def __init__(
        self, h: int, hv: int, k_dim: int = 128, v_dim: int = 128, cache_len: int = 16
    ) -> None:
        assert hv % h == 0
        assert hv // h <= 4
        assert k_dim == 128 and v_dim == 128 and (cache_len == 16)
        self.h = h
        self.hv = hv
        self.v_dim = v_dim
        self.cache_len = cache_len
        self.num_warps = 6
        self.mma_n = 16
        self.elems = 32
        self.n_groups = k_dim // self.elems

    @cute.jit
    def _make_state_tma(self, state: cute.Tensor, op: cpasync.TmaCopyOp):
        num_elems = 128 // (state.element_type.width // 8)
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
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        stream: CUstream,
    ):
        state_tma = self._make_state_tma(state, cpasync.CopyBulkTensorTileG2SOp())
        self.kernel(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            d_cache,
            k_cache,
            g_cache,
            state_indices,
            write_pos,
            out,
            scale,
            state_tma,
        ).launch(
            grid=(q.shape[0], self.h, 1),
            block=(self.num_warps * 32, 1, 1),
            min_blocks_per_mp=5,
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
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        state_tma: cpasync.TmaInfo,
    ):
        state_t = Float32
        cache_t = Float16
        elems = self.elems
        n_groups = self.n_groups
        tid, _, _ = cute.arch.thread_idx()
        batch_id, key_head, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        heads_per_key = self.hv // self.h
        first_value_head = key_head * heads_per_key
        n_slots = heads_per_key * n_groups

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        state_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (self.v_dim, (elems, 2)), stride=(elems, (1, self.v_dim * elems))
            ),
        )
        s_state = allocate_tensor(smem, state_t, state_layout)
        qk_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (self.mma_n, (elems, n_groups)), stride=(elems, (1, self.mma_n * elems))
            ),
        )
        s_qk = allocate_tensor(smem, state_t, qk_layout)
        s_norm_partial = smem.allocate_array(Float32, 12)
        s_stats = smem.allocate_array(Float32, 3)
        s_cq = smem.allocate_array(Float32, self.cache_len)
        s_ck = smem.allocate_array(Float32, self.cache_len)
        s_coef_q = smem.allocate_array(Float32, self.cache_len * 4)
        s_coef_k = smem.allocate_array(Float32, self.cache_len * 4)
        s_gate = smem.allocate_array(Float32, 16)
        qk_full = smem.allocate_array(Int64, 1)
        state_slot = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        mma_slot = [smem.allocate_array(Int64, 1) for _ in range(n_slots)]
        epilogue_done = smem.allocate_array(Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)
        state_idx = state_indices[batch_id]
        padded_row = tid - 64
        padded_out_ptrs = [
            out.iterator
            + cute.crd2idx(
                (batch_id, first_value_head + min(i, heads_per_key - 1), padded_row),
                out.layout,
            )
            for i in range(4)
        ]
        _zero_output_and_exit_if_invalid(state_idx, tid, *padded_out_ptrs)
        position_tag = write_pos[batch_id]
        wp = Int32(position_tag)
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(state_slot[0], 1)
                cute.arch.mbarrier_init(state_slot[1], 1)
                cute.arch.mbarrier_init_fence()
            cpasync.prefetch_descriptor(state_tma.atom)
            for g in cutlass.range_constexpr(2):
                g_state = state_tma.tma_tensor[
                    state_idx, first_value_head, None, (None, g)
                ]
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(state_slot[g], 16384)
                simple_tma_copy(
                    state_tma.atom, g_state, s_state[None, (None, g)], state_slot[g]
                )
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(qk_full, 128)
                for slot in cutlass.range_constexpr(2, n_slots):
                    cute.arch.mbarrier_init(state_slot[slot], 1)
                for slot in cutlass.range_constexpr(n_slots):
                    cute.arch.mbarrier_init(mma_slot[slot], 1)
                cute.arch.mbarrier_init(epilogue_done, 128 * heads_per_key)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp_id == 0:
            for slot in cutlass.range_constexpr(2, n_slots):
                value_head = first_value_head + slot // n_groups
                g = slot % n_groups
                cute.arch.mbarrier_wait(mma_slot[slot - 2], 0)
                g_state = state_tma.tma_tensor[state_idx, value_head, None, (None, g)]
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(state_slot[slot], 16384)
                simple_tma_copy(
                    state_tma.atom,
                    g_state,
                    s_state[None, (None, slot % 2)],
                    state_slot[slot],
                )
        elif warp_id == 1:
            _alloc_tmem(tmem_addr, 64)
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_base = tmem_addr[0]
            cute.arch.mbarrier_wait(qk_full, 0)
            sdesc = _tcgen05.make_sdesc_128B_swizzle(self.v_dim * 128)
            qk_base = sdesc | s_qk.iterator.toint() >> 4
            state_base0 = sdesc | s_state.iterator.toint() >> 4
            state_base1 = sdesc | s_state.iterator.toint() + 16384 >> 4
            idesc = _make_tf32_idesc(self.v_dim, self.mma_n)
            for head_offset in cutlass.range_constexpr(heads_per_key):
                for g in cutlass.range_constexpr(n_groups):
                    slot = head_offset * n_groups + g
                    cute.arch.mbarrier_wait(state_slot[slot], 0)
                    _tcgen05.fence_after_thread_sync()
                    stage_base = state_base0 if slot % 2 == 0 else state_base1
                    for k_atom in cutlass.range_constexpr(4):
                        state_desc = stage_base | k_atom * 32 >> 4
                        qk_desc = qk_base | g * self.mma_n * 128 + k_atom * 32 >> 4
                        _mma_tf32(
                            tmem_base + head_offset * self.mma_n,
                            state_desc,
                            qk_desc,
                            idesc,
                            g > 0 or k_atom > 0,
                        )
                    _tcgen05.commit(mma_slot[slot])
            cute.arch.mbarrier_wait(epilogue_done, 0)
            _dealloc_tmem(tmem_addr[0], 64)
        else:
            consumer_warp = warp_id - 2
            k_idx = consumer_warp * 32 + lane_id
            q_raw = Float32(q[batch_id, key_head, k_idx])
            k_raw = Float32(k[batch_id, key_head, k_idx])
            q_sq = cute.arch.warp_reduction_sum(q_raw * q_raw)
            k_sq = cute.arch.warp_reduction_sum(k_raw * k_raw)
            qk_dot = cute.arch.warp_reduction_sum(q_raw * k_raw)
            if lane_id == 0:
                s_norm_partial[consumer_warp * 3] = q_sq
                s_norm_partial[consumer_warp * 3 + 1] = k_sq
                s_norm_partial[consumer_warp * 3 + 2] = qk_dot
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            if consumer_warp == 0 and lane_id == 0:
                q_total = Float32(0.0)
                k_total = Float32(0.0)
                qk_total = Float32(0.0)
                for part in cutlass.range_constexpr(4):
                    q_total += s_norm_partial[part * 3]
                    k_total += s_norm_partial[part * 3 + 1]
                    qk_total += s_norm_partial[part * 3 + 2]
                q_rnorm = cute.math.rsqrt(q_total + Float32(1e-06))
                k_rnorm = cute.math.rsqrt(k_total + Float32(1e-06))
                s_stats[0] = q_rnorm
                s_stats[1] = k_rnorm
                s_stats[2] = qk_total * q_rnorm * k_rnorm * scale
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            q_scaled = q_raw * s_stats[0] * scale
            k_norm = k_raw * s_stats[1]
            s_qk[0, (k_idx % elems, k_idx // elems)] = state_t(q_scaled)
            s_qk[1, (k_idx % elems, k_idx // elems)] = state_t(k_norm)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(qk_full)
            k_cache[state_idx, key_head, wp, k_idx] = cache_t(k_norm)
            cute.arch.barrier(barrier_id=6, number_of_threads=128)
            for cache_group in cutlass.range_constexpr(4):
                cache_idx = consumer_warp + cache_group * 4
                cq = Float32(0.0)
                ck = Float32(0.0)
                if cache_idx < wp:
                    for k_group in cutlass.range_constexpr(4):
                        kk = lane_id + k_group * 32
                        cached_k = Float32(k_cache[state_idx, key_head, cache_idx, kk])
                        cq += cached_k * Float32(s_qk[0, (kk % elems, kk // elems)])
                        ck += cached_k * Float32(s_qk[1, (kk % elems, kk // elems)])
                cq = cute.arch.warp_reduction_sum(cq)
                ck = cute.arch.warp_reduction_sum(ck)
                if lane_id == 0:
                    s_cq[cache_idx] = cq
                    s_ck[cache_idx] = ck
            cute.arch.barrier(barrier_id=3, number_of_threads=128)
            tmem_row = consumer_warp * 32 + lane_id
            row = tmem_row ^ 64
            for head_offset in cutlass.range_constexpr(heads_per_key):
                value_head = first_value_head + head_offset
                if consumer_warp == 0:
                    cached_g = Float32(0.0)
                    if lane_id < self.cache_len and lane_id < wp:
                        cached_g = g_cache[state_idx, value_head, lane_id]
                    g_prefix = cached_g
                    for offset in (1, 2, 4, 8):
                        lower = cute.arch.shuffle_sync_up(
                            g_prefix, offset, mask_and_clamp=0
                        )
                        if lane_id >= offset:
                            g_prefix += lower
                    g_total = cute.arch.shuffle_sync(g_prefix, 15)
                    if lane_id < self.cache_len:
                        replay = Float32(0.0)
                        if lane_id < wp:
                            replay = cute.math.exp(g_total - g_prefix, fastmath=True)
                        coef_idx = head_offset * self.cache_len + lane_id
                        s_coef_q[coef_idx] = replay * s_cq[lane_id]
                        s_coef_k[coef_idx] = replay * s_ck[lane_id]
                    if lane_id == 0:
                        x = Float32(a[batch_id, value_head]) + Float32(
                            dt_bias[value_head]
                        )
                        softplus = x
                        if x <= Float32(20.0):
                            softplus = cute.math.log(
                                Float32(1.0) + cute.math.exp(x, fastmath=True),
                                fastmath=True,
                            )
                        g_new = (
                            -cute.math.exp(Float32(a_log[value_head]), fastmath=True)
                            * softplus
                        )
                        alpha = cute.math.exp(g_new, fastmath=True)
                        beta = Float32(
                            BFloat16(
                                Float32(1.0)
                                / (
                                    Float32(1.0)
                                    + cute.math.exp(
                                        -Float32(b[batch_id, value_head]), fastmath=True
                                    )
                                )
                            )
                        )
                        s_gate[head_offset * 4] = g_new
                        s_gate[head_offset * 4 + 1] = alpha
                        s_gate[head_offset * 4 + 2] = beta
                        s_gate[head_offset * 4 + 3] = cute.math.exp(
                            g_total, fastmath=True
                        )
                        g_cache[state_idx, value_head, wp] = g_new
                cute.arch.barrier(barrier_id=4, number_of_threads=128)
                d_q = Float32(0.0)
                d_k = Float32(0.0)
                for cache_idx in cutlass.range_constexpr(self.cache_len):
                    cached_d = Float32(d_cache[state_idx, value_head, cache_idx, row])
                    d_q += cached_d * s_coef_q[head_offset * self.cache_len + cache_idx]
                    d_k += cached_d * s_coef_k[head_offset * self.cache_len + cache_idx]
                v_val = Float32(v[batch_id, value_head, row])
                alpha = s_gate[head_offset * 4 + 1]
                beta = s_gate[head_offset * 4 + 2]
                total_decay = s_gate[head_offset * 4 + 3]
                cute.arch.mbarrier_wait(
                    mma_slot[head_offset * n_groups + n_groups - 1], 0
                )
                _tcgen05.fence_after_thread_sync()
                tmem_base = tmem_addr[0]
                q_readout = _tcgen05.ld(
                    tmem_row, tmem_base + head_offset * self.mma_n, "32x32b", 1
                )
                k_readout = _tcgen05.ld(
                    tmem_row, tmem_base + head_offset * self.mma_n + 1, "32x32b", 1
                )
                _tcgen05.wait_ld()
                sq = total_decay * q_readout + d_q
                sk = total_decay * k_readout + d_k
                d_cur = beta * (v_val - alpha * sk)
                output = alpha * sq + d_cur * s_stats[2]
                out[batch_id, value_head, row] = BFloat16(output)
                d_cache[state_idx, value_head, wp, row] = cache_t(d_cur)
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(epilogue_done)

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
        state = make_fake_tensor(Float32, (slots, hv, v_dim, k_dim), divisibility=16)
        ring_t = Float16
        d_cache = make_fake_tensor(
            ring_t, (slots, hv, cache_len, v_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            ring_t, (slots, h, cache_len, k_dim), divisibility=16
        )
        g_cache = make_fake_tensor(Float32, (slots, hv, cache_len), divisibility=16)
        state_indices = make_fake_tensor(Int32, (batch,), divisibility=1)
        write_pos = make_fake_tensor(Int32, (batch,), divisibility=1)
        out = make_fake_tensor(BFloat16, (batch, hv, v_dim), divisibility=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm103Fp16ReplayFusedKernel(h, hv, k_dim, v_dim, cache_len),
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state,
            d_cache,
            k_cache,
            g_cache,
            state_indices,
            write_pos,
            out,
            Float32(1.0),
            stream,
            options="--enable-tvm-ffi",
        )


def gdn_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _, h, k_dim = q.shape
    _, hv, v_dim = v.shape
    cache_len = d_cache.shape[2]
    if state.dtype != torch.float32:
        raise TypeError("state must be torch.float32")
    if d_cache.dtype != torch.float16 or k_cache.dtype != torch.float16:
        raise TypeError("d/k ring caches must both be torch.float16")
    if write_pos.dtype != torch.int32:
        raise TypeError("write_pos must be int32")
    if a_log.dtype != torch.float32:
        a_log = a_log.float()
    if dt_bias.dtype != torch.float32:
        dt_bias = dt_bias.float()
    Sm103Fp16ReplayFusedKernel.compile(h, hv, k_dim, v_dim, cache_len)(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        d_cache,
        k_cache,
        g_cache,
        state_indices,
        write_pos,
        out,
        float(scale),
    )
    return out
