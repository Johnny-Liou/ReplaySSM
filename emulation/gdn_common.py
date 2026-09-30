# SPDX-License-Identifier: Apache-2.0
"""GDN normalization and window selection."""

from __future__ import annotations

import triton
import triton.language as tl

NORM_EPS = tl.constexpr(1e-06)


@triton.jit
def _row_l2norm(m, eps: tl.constexpr):
    return m / tl.sqrt(tl.sum(m * m, axis=1) + eps)[:, None]


@triton.jit
def _pick(vec, idx, offs):
    return tl.sum(tl.where(offs == idx, vec, 0.0))
