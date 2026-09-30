# SPDX-License-Identifier: Apache-2.0
"""Fixed-batch decode examples with device-side window counters."""

import torch
import triton

from .gdn_replay_unf_vf import _gdn_unf_vf_step_kernel
from .mamba2_replay import _flush_quant_kernel, _replay_step_kernel
from .modes import MODES


class _Replay:
    def __init__(self, state, mode, tile):
        if state.ndim != 4 or not state.is_cuda or state.dtype != torch.float32:
            raise ValueError(
                "state must be a CUDA FP32 tensor [batch, heads, rows, columns]"
            )
        if (
            any(d <= 0 for d in state.shape)
            or state.shape[-1] < 128
            or not triton.next_power_of_2(state.shape[-1]) == state.shape[-1]
        ):
            raise ValueError(
                "dimensions must be positive; state columns must be a power of two >= 128"
            )
        try:
            self.precision = MODES[mode]
        except KeyError:
            raise ValueError(f"mode must be one of {tuple(MODES)}") from None
        self.state_hi = state.clone().contiguous()
        self.state_lo = state.clone().contiguous()
        self.batch, self.heads, self.rows, self.columns = state.shape
        self.tile = tile
        self.tiles = triton.next_power_of_2(triton.cdiv(self.rows, tile))
        self.amax = self._zeros(self.batch, self.heads, self.tiles)
        self.steps = self._zeros(self.batch, dtype=torch.int32)
        self.write_pos = self._zeros(self.batch, dtype=torch.int32)
        self.window_idx = self._zeros(self.batch, dtype=torch.int32)
        self.is_flush = self._zeros(self.batch, dtype=torch.int8)
        self.state_idx = torch.arange(
            self.batch, device=state.device, dtype=torch.int32
        )
        self.output = self._zeros(self.batch, self.heads, self.rows)

    def _zeros(self, *shape, dtype=torch.float32):
        return torch.zeros(shape, device=self.state_hi.device, dtype=dtype)

    def _prepare_step(self):
        torch.remainder(self.steps, 16, out=self.write_pos)
        torch.div(self.steps, 16, rounding_mode="floor", out=self.window_idx)
        torch.eq(self.write_pos, 15, out=self.is_flush)

    def _finish_step(self, grid):
        p = self.precision
        if p.nvfp4:
            _flush_quant_kernel[grid](
                self.state_hi,
                self.state_lo,
                self.amax,
                self.write_pos,
                self.is_flush,
                self.window_idx,
                self.state_idx,
                -1,
                1234,
                self.state_hi.stride(0),
                self.amax.stride(0),
                self.batch,
                self.heads,
                self.rows,
                self.columns,
                self.tile,
                self.tiles,
                p.two_precision,
                p.state,
                p.stochastic,
                p.read if p.two_precision else p.state,
            )
        self.steps.add_(1)
        return self.output

    def _check_inputs(self, tensors):
        for tensor, shape, dtype in tensors:
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self.state_hi.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"expected contiguous {dtype} tensor of shape {shape} on {self.state_hi.device}"
                )


class Mamba2Replay(_Replay):
    """Decode from an FP32 post-prefill state [batch, heads, head_dim, dstate]."""

    def __init__(self, state, num_groups, mode="tp4_hi16_sr"):
        super().__init__(state, mode, tile=16)
        if num_groups <= 0 or self.heads % num_groups:
            raise ValueError("heads must be divisible by num_groups")
        self.groups = num_groups
        self.x_cache = self._zeros(
            self.batch, self.heads, 16, self.rows, dtype=torch.bfloat16
        )
        self.dt_cache = self._zeros(self.batch, self.heads, 16)
        self.B_cache = self._zeros(
            self.batch, self.groups, 16, self.columns, dtype=torch.bfloat16
        )

    def step(self, x, dt, B, C, A, D, dt_bias):
        """Advance one token. The returned FP32 output buffer is reused each step."""
        b, h, p, n, g = self.batch, self.heads, self.rows, self.columns, self.groups
        self._check_inputs(
            [
                (x, (b, h, p), torch.bfloat16),
                (dt, (b, h), torch.float32),
                (B, (b, g, n), torch.bfloat16),
                (C, (b, g, n), torch.bfloat16),
                (A, (h,), torch.float32),
                (D, (h,), torch.float32),
                (dt_bias, (h,), torch.float32),
            ]
        )
        self._prepare_step()
        prec = self.precision
        grid = (triton.cdiv(p, self.tile), b, h)
        _replay_step_kernel[grid](
            self.state_hi,
            self.state_lo,
            self.amax,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            self.output,
            self.x_cache,
            self.dt_cache,
            self.B_cache,
            self.write_pos,
            self.is_flush,
            self.window_idx,
            self.state_idx,
            -1,
            1234,
            x.stride(0),
            dt.stride(0),
            B.stride(0),
            C.stride(0),
            self.output.stride(0),
            self.state_hi.stride(0),
            self.x_cache.stride(0),
            self.dt_cache.stride(0),
            self.B_cache.stride(0),
            self.amax.stride(0),
            b,
            h,
            p,
            n,
            g,
            self.tile,
            self.tiles,
            prec.two_precision,
            prec.state,
            prec.stochastic,
            prec.read if prec.two_precision else prec.state,
            prec.nvfp4,
        )
        return self._finish_step(grid)


class GDNReplay(_Replay):
    """Decode from an FP32 post-prefill state [batch, value_heads, value_dim, key_dim]."""

    def __init__(self, state, key_heads, mode="tp4_hi16_sr"):
        super().__init__(state, mode, tile=64)
        if key_heads <= 0 or self.heads % key_heads:
            raise ValueError("value_heads must be divisible by key_heads")
        self.key_heads = key_heads
        self.k_cache = self._zeros(
            self.batch, key_heads, 16, self.columns, dtype=torch.bfloat16
        )
        self.v_cache = self._zeros(
            self.batch, self.heads, 16, self.rows, dtype=torch.bfloat16
        )
        self.g_cache = self._zeros(self.batch, self.heads, 16)
        self.beta_cache = self._zeros(self.batch, self.heads, 16)
        self.u_cache = self._zeros(
            self.batch, self.heads, 16, self.rows, dtype=torch.float16
        )

    def step(self, qkv, a, b, A_log, dt_bias):
        """Advance one token from packed [q, k, v]; the output buffer is reused."""
        r, h, hv, k, v = self.batch, self.key_heads, self.heads, self.columns, self.rows
        self._check_inputs(
            [
                (qkv, (r, 2 * h * k + hv * v), torch.bfloat16),
                (a, (r, hv), torch.float32),
                (b, (r, hv), torch.float32),
                (A_log, (hv,), torch.float32),
                (dt_bias, (hv,), torch.float32),
            ]
        )
        self._prepare_step()
        p = self.precision
        grid = (triton.cdiv(v, self.tile), r, hv)
        _gdn_unf_vf_step_kernel[grid](
            self.state_hi,
            self.state_lo,
            self.amax,
            qkv,
            a,
            b,
            A_log,
            dt_bias,
            self.output,
            self.k_cache,
            self.v_cache,
            self.g_cache,
            self.beta_cache,
            self.u_cache,
            self.write_pos,
            self.is_flush,
            self.window_idx,
            self.state_idx,
            -1,
            1234,
            k**-0.5,
            qkv.stride(0),
            a.stride(0),
            b.stride(0),
            self.output.stride(0),
            self.state_hi.stride(0),
            self.k_cache.stride(0),
            self.v_cache.stride(0),
            self.g_cache.stride(0),
            self.beta_cache.stride(0),
            self.u_cache.stride(0),
            self.amax.stride(0),
            r,
            h,
            hv,
            k,
            v,
            self.tile,
            self.tiles,
            p.two_precision,
            p.state,
            p.stochastic,
            p.read if p.two_precision else p.state,
            p.nvfp4,
        )
        return self._finish_step(grid)
