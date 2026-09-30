# SPDX-License-Identifier: Apache-2.0
"""GDN fp16 nonflush; fixed L=16 specialization."""

from __future__ import annotations

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float16, Float32, Int8, Int32, Int64, Uint64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op
from quack.compile_utils import make_fake_tensor

from low_precision.common import _tcgen05, simple_tma_copy
from low_precision.common.gdn_ops import (
    _alloc_tmem,
    _dealloc_tmem,
    _make_fp16_idesc,
    _zero_output_and_exit_if_invalid,
)
from low_precision.common.nf_ops import cp_async_bulk_g2s


@dsl_user_op
def _exit_if_nonzero(flag: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(flag).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $0, 0;\n\t@p exit;\n\t}",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
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
        self.elems = 64
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
        v_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
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
            v_cache,
            beta_cache,
            state_indices,
            write_pos,
            is_flush,
            out,
            scale,
            state_tma,
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
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        v_cache: cute.Tensor,
        beta_cache: cute.Tensor,
        state_indices: cute.Tensor,
        write_pos: cute.Tensor,
        is_flush: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        state_tma: cpasync.TmaInfo,
    ):
        state_t = Float16
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
        s_grow = smem.allocate_tensor(
            Int8, cute.make_layout(heads_per_key * 64), byte_alignment=128
        )
        g_full = smem.allocate_array(Int64, 1)
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
        _exit_if_nonzero(Int32(is_flush[batch_id]))
        position_tag = write_pos[batch_id]
        position_tag_u64 = Uint64(position_tag)
        wp = Int32(position_tag_u64 & Uint64(self.cache_len - 1))
        grow_s = s_grow.iterator.toint()
        grow_g = g_cache.iterator.toint() + Int64(
            state_idx * self.hv + first_value_head
        ) * Int64(self.cache_len * 4)
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(state_slot[0], 1)
                cute.arch.mbarrier_init(state_slot[1], 1)
                cute.arch.mbarrier_init(g_full, 1)
                cute.arch.mbarrier_init_fence()
            if lane_id == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(g_full, heads_per_key * 64)
                cp_async_bulk_g2s(grow_s, grow_g, heads_per_key * 64, g_full.toint())
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
            idesc = _make_fp16_idesc(self.v_dim, self.mma_n)
            for head_offset in cutlass.range_constexpr(heads_per_key):
                for g in cutlass.range_constexpr(n_groups):
                    slot = head_offset * n_groups + g
                    cute.arch.mbarrier_wait(state_slot[slot], 0)
                    _tcgen05.fence_after_thread_sync()
                    stage_base = state_base0 if slot % 2 == 0 else state_base1
                    for k_atom in cutlass.range_constexpr(4):
                        state_desc = stage_base | k_atom * 32 >> 4
                        qk_desc = qk_base | g * self.mma_n * 128 + k_atom * 32 >> 4
                        _tcgen05.mma_f16(
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
            k_cache[state_idx, key_head, wp, k_idx] = BFloat16(k_raw)
            cute.arch.barrier(barrier_id=6, number_of_threads=128)
            for cache_group in cutlass.range_constexpr(4):
                cache_idx = consumer_warp + cache_group * 4
                cq = Float32(0.0)
                ck = Float32(0.0)
                kn = Float32(0.0)
                if cache_idx < wp:
                    for k_group in cutlass.range_constexpr(4):
                        kk = lane_id + k_group * 32
                        cached_k = Float32(k_cache[state_idx, key_head, cache_idx, kk])
                        cq += cached_k * Float32(s_qk[0, (kk % elems, kk // elems)])
                        ck += cached_k * Float32(s_qk[1, (kk % elems, kk // elems)])
                        kn += cached_k * cached_k
                cq = cute.arch.warp_reduction_sum(cq)
                ck = cute.arch.warp_reduction_sum(ck)
                kn = cute.arch.warp_reduction_sum(kn)
                if lane_id == 0:
                    rn = cute.math.rsqrt(kn + Float32(1e-06))
                    s_cq[cache_idx] = cq * rn
                    s_ck[cache_idx] = ck * rn
            cute.arch.barrier(barrier_id=3, number_of_threads=128)
            tmem_row = consumer_warp * 32 + lane_id
            row = tmem_row ^ 64
            prep_head_offset = consumer_warp
            prep_value_head = first_value_head + prep_head_offset
            if prep_head_offset < heads_per_key:
                cached_g = Float32(0.0)
                if lane_id < self.cache_len and lane_id < wp:
                    cached_g = g_cache[state_idx, prep_value_head, lane_id]
                a_value = Float32(0.0)
                dt_bias_value = Float32(0.0)
                a_log_value = Float32(0.0)
                b_value = Float32(0.0)
                if lane_id == 0:
                    a_value = Float32(a[batch_id, prep_value_head])
                    dt_bias_value = Float32(dt_bias[prep_value_head])
                    a_log_value = Float32(a_log[prep_value_head])
                    b_value = Float32(b[batch_id, prep_value_head])
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
                    coef_idx = prep_head_offset * self.cache_len + lane_id
                    s_coef_q[coef_idx] = replay * s_cq[lane_id]
                    s_coef_k[coef_idx] = replay * s_ck[lane_id]
                if lane_id == 0:
                    x = a_value + dt_bias_value
                    softplus = x
                    if x <= Float32(20.0):
                        softplus = cute.math.log(
                            Float32(1.0) + cute.math.exp(x, fastmath=True),
                            fastmath=True,
                        )
                    g_new = -cute.math.exp(a_log_value, fastmath=True) * softplus
                    alpha = cute.math.exp(g_new, fastmath=True)
                    beta = Float32(
                        BFloat16(
                            Float32(1.0)
                            / (Float32(1.0) + cute.math.exp(-b_value, fastmath=True))
                        )
                    )
                    gate_idx = prep_head_offset * 4
                    s_gate[gate_idx] = g_new
                    s_gate[gate_idx + 1] = alpha
                    s_gate[gate_idx + 2] = beta
                    s_gate[gate_idx + 3] = cute.math.exp(g_total, fastmath=True)
                    g_cache[state_idx, prep_value_head, wp] = g_new
                    beta_cache[state_idx, prep_value_head, wp] = beta
            cute.arch.barrier(barrier_id=4, number_of_threads=128)
            cute.arch.mbarrier_wait(g_full, 0)
            u_pre = []
            for cache_idx in cutlass.range_constexpr(self.cache_len):
                u_pre.append(
                    Float32(d_cache[state_idx, first_value_head, cache_idx, row])
                )
            v_pre = Float32(v[batch_id, first_value_head, row])
            for head_offset in cutlass.range_constexpr(heads_per_key):
                value_head = first_value_head + head_offset
                d_q = Float32(0.0)
                d_k = Float32(0.0)
                for cache_idx in cutlass.range_constexpr(self.cache_len):
                    cached_d = u_pre[cache_idx]
                    if cache_idx >= wp:
                        cached_d = Float32(0.0)
                    d_q += cached_d * s_coef_q[head_offset * self.cache_len + cache_idx]
                    d_k += cached_d * s_coef_k[head_offset * self.cache_len + cache_idx]
                v_val = v_pre
                if cutlass.const_expr(head_offset + 1 < heads_per_key):
                    u_pre = []
                    for cache_idx in cutlass.range_constexpr(self.cache_len):
                        u_pre.append(
                            Float32(d_cache[state_idx, value_head + 1, cache_idx, row])
                        )
                    v_pre = Float32(v[batch_id, value_head + 1, row])
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
                v_cache[state_idx, value_head, wp, row] = BFloat16(v_val)
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
        state = make_fake_tensor(Float16, (slots, hv, v_dim, k_dim), divisibility=16)
        ring_t = Float16
        d_cache = make_fake_tensor(
            ring_t, (slots, hv, cache_len, v_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            BFloat16, (slots, h, cache_len, k_dim), divisibility=16
        )
        g_cache = make_fake_tensor(Float32, (slots, hv, cache_len), divisibility=16)
        v_cache = make_fake_tensor(
            BFloat16, (slots, hv, cache_len, v_dim), divisibility=16
        )
        beta_cache = make_fake_tensor(Float32, (slots, hv, cache_len), divisibility=16)
        is_flush = make_fake_tensor(Int8, (batch,), divisibility=1)
        state_indices = make_fake_tensor(Int32, (batch,), divisibility=1)
        write_pos = make_fake_tensor(Int64, (batch,), divisibility=1)
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
            v_cache,
            beta_cache,
            state_indices,
            write_pos,
            is_flush,
            out,
            Float32(1.0),
            stream,
            options="--enable-tvm-ffi",
        )


def gdn_fp16_nonflush(
    q,
    k,
    v,
    a,
    b,
    a_log,
    dt_bias,
    scale,
    state,
    u_cache,
    k_cache,
    g_cache,
    v_cache,
    beta_cache,
    state_indices,
    write_pos,
    is_flush,
    out,
) -> torch.Tensor:
    _, h, k_dim = q.shape
    _, hv, v_dim = v.shape
    cache_len = u_cache.shape[2]
    if state.dtype != torch.float16:
        raise TypeError("state must be torch.float16")
    if (
        u_cache.dtype != torch.float16
        or k_cache.dtype != torch.bfloat16
        or v_cache.dtype != torch.bfloat16
    ):
        raise TypeError("u ring fp16, k/v rings bf16")
    if write_pos.dtype != torch.int64:
        raise TypeError("stochastic rounding requires int64 packed position tags")
    Sm103Fp16ReplayFusedKernel.compile(h, hv, k_dim, v_dim, cache_len)(
        q,
        k,
        v,
        a,
        b,
        a_log.float(),
        dt_bias.float(),
        state,
        u_cache,
        k_cache,
        g_cache,
        v_cache,
        beta_cache,
        state_indices,
        write_pos,
        is_flush.view(torch.int8),
        out,
        float(scale),
    )
    return out
