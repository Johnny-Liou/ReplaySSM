# SPDX-License-Identifier: Apache-2.0
"""Shared CuTe device operations."""

from __future__ import annotations

from cutlass import Boolean, Float32, Int32, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

from low_precision.common import _tcgen05

_PHILOX_ROUNDS_DEFAULT = 7
_PHILOX_ROUND_A = 3528531795
_PHILOX_ROUND_B = 3449720151
_PHILOX_KEY_A = 2654435769
_PHILOX_KEY_B = 3144134277


@dsl_user_op
def _mul_wide_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> tuple[Uint32, Uint32]:
    result_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    result = llvm.inline_asm(
        result_type,
        [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
        (
            "{\n"
            "\t.reg .u64 product;\n"
            "\tmul.wide.u32 product, $2, $3;\n"
            "\tmov.b64 {$1, $0}, product;\n"
            "\t}"
        ),
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    word_type = T.i32()
    hi = Uint32(llvm.extractvalue(word_type, result, [0], loc=loc, ip=ip))
    lo = Uint32(llvm.extractvalue(word_type, result, [1], loc=loc, ip=ip))
    return (hi, lo)


@dsl_user_op
def _philox4x32(
    counter0: Uint32,
    counter1: Uint32,
    counter2: Uint32,
    counter3: Uint32,
    key0: Uint32,
    key1: Uint32,
    n_rounds: int = _PHILOX_ROUNDS_DEFAULT,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    c0 = Uint32(counter0)
    c1 = Uint32(counter1)
    c2 = Uint32(counter2)
    c3 = Uint32(counter3)
    k0 = Uint32(key0)
    k1 = Uint32(key1)
    round_a = Uint32(_PHILOX_ROUND_A)
    round_b = Uint32(_PHILOX_ROUND_B)
    key_a = Uint32(_PHILOX_KEY_A)
    key_b = Uint32(_PHILOX_KEY_B)
    for _ in range(n_rounds):
        hi_b, lo_b = _mul_wide_u32(c2, round_b, loc=loc, ip=ip)
        hi_a, lo_a = _mul_wide_u32(c0, round_a, loc=loc, ip=ip)
        c0 = hi_b ^ c1 ^ k0
        c2 = hi_a ^ c3 ^ k1
        c1 = lo_b
        c3 = lo_a
        k0 = k0 + key_a
        k1 = k1 + key_b
    return (c0, c1, c2, c3)


@dsl_user_op
def _cvt_f32x2_f16x2_rs(
    a: Float32, b: Float32, rand_bits: Uint32, *, loc=None, ip=None
) -> Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Uint32(rand_bits).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rs.satfinite.f16x2.f32 $0, $2, $1, $3;",
        "=r,f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _alloc_tmem(taddr: cute.Pointer, ncols: int, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_alloc(
        taddr.to_llvm_ptr(loc=loc, ip=ip),
        Uint32(ncols).ir_value(loc=loc, ip=ip),
        group=nvvm.Tcgen05GroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _dealloc_tmem(tmem_addr: Int32, ncols: int, *, loc=None, ip=None) -> None:
    ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    tmem_ptr = llvm.inttoptr(
        ptr_ty, Int32(tmem_addr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    nvvm.tcgen05_dealloc(
        tmem_ptr,
        Int32(ncols).ir_value(loc=loc, ip=ip),
        group=nvvm.Tcgen05GroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _zero_output_and_exit_if_invalid(
    state_idx: Int32,
    tid: Int32,
    out0: cute.Pointer,
    out1: cute.Pointer,
    out2: cute.Pointer,
    out3: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(state_idx).ir_value(loc=loc, ip=ip),
            Int32(tid).ir_value(loc=loc, ip=ip),
            *[ptr.toint(loc=loc, ip=ip).ir_value() for ptr in (out0, out1, out2, out3)],
        ],
        (
            "{\n"
            "\t.reg .pred invalid, consumer, store_zero;\n"
            "\tsetp.le.s32 invalid, $0, 0;\n"
            "\tsetp.ge.u32 consumer, $1, 64;\n"
            "\tand.pred store_zero, invalid, consumer;\n"
            "\t@store_zero st.global.u16 [$2], 0;\n"
            "\t@store_zero st.global.u16 [$3], 0;\n"
            "\t@store_zero st.global.u16 [$4], 0;\n"
            "\t@store_zero st.global.u16 [$5], 0;\n"
            "\t@invalid exit;\n"
            "\t}"
        ),
        "r,r,l,l,l,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


def _make_fp16_idesc(mma_m: int, mma_n: int) -> Uint32:
    return Uint32(1 << 4 | mma_n >> 3 << 17 | mma_m >> 4 << 24)


def _make_sdesc_32B_swizzle(base16) -> Uint64:
    half_mode = Uint64(3 << 61)
    lo = Uint64(1 << 16 | 16 << 32 | 1 << 46)
    return base16 + half_mode + half_mode | lo


def _make_tf32_idesc(mma_m: int, mma_n: int) -> Uint32:
    return Uint32(1 << 4 | 2 << 7 | 2 << 10 | mma_n >> 3 << 17 | mma_m >> 4 << 24)


@dsl_user_op
def _mma_tf32(
    d_tmem, a_desc, b_desc, idesc, enable_input_d, *, loc=None, ip=None
) -> None:
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.TF32,
            nvvm.Tcgen05GroupKind.CTA_1,
            _tcgen05._make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
