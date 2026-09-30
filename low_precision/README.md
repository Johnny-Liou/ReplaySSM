# Low-precision CuTe kernels

Standard decoding kernels for **B300 / SM103**, with window length 16. These
kernels support the latency benefit of low-precision checkpoint reads enabled by
ReplaySSM's separation of output computation and state updates.

They store compressed read copies and an FP16 recurrence anchor. Every anchor
flush uses stochastic rounding; each read copy is quantized independently from
the unrounded FP32 refreshed state. These kernels are standalone and are not
wired into the vLLM serving dispatch.

| Precision | Mamba2 | GDN |
| --- | --- | --- |
| FP16 + SR | `mamba2/fp16.py` (non-flush and flush) | `gdn/fp16_nonflush.py`, `gdn/fp16_flush.py` |
| FP8 read, FP16 + SR anchor | `mamba2/fp8_nonflush.py`, `mamba2/fp8_flush.py` | `gdn/fp8_nonflush.py`, `gdn/fp8_flush.py` |
| NVFP4 read, FP16 + SR anchor | `mamba2/nvfp4_nonflush.py`, `mamba2/nvfp4_flush.py` | `gdn/nvfp4_nonflush.py`, `gdn/nvfp4_flush.py` |
| FP32 non-flush comparator | `mamba2/fp32.py` | `gdn/fp32.py` |

GDN uses `unf_vf`: FP16 update rings for non-flush outputs and raw BF16 key/value
rings for flush reconstruction. The FP32 GDN comparator uses a d/k input ring.

The Mamba2 FP8 readout accumulates in FP32. GDN FP8 and both NVFP4
readouts use packed FP16 accumulation. FP8 uses block-128 FP32 scales. NVFP4 uses
block-16 E4M3 scales and a request-wide FP32 outer scale. Its flush uses maxima
from the preceding checkpoint to select that outer scale, while recording new
maxima for the next flush. This avoids a cross-CTA reduction of the current state.

## Setup and checks

Tested with Python 3.12, PyTorch 2.13.0+cu130 and the public CUTLASS DSL 4.5.2 wheel.
Use an environment separate from the vLLM build:

```bash
uv venv low_precision/.venv --python 3.12
uv pip install --python low_precision/.venv/bin/python torch==2.13.0 --torch-backend=cu130
uv pip install --python low_precision/.venv/bin/python -r low_precision/requirements.txt pytest
low_precision/.venv/bin/python -m pytest low_precision/tests/test_decode.py -q
```

FP16 stochastic conversion instructions require Blackwell. Importing the package
raises the preferred shared-memory carveout to 100% for every kernel that requests
at least three blocks per SM, the setting the kernels were tuned with.

## Calling the kernels

Each module exposes a launch function named after its path, so
`mamba2/fp8_flush.py` exposes `mamba2_fp8_flush`. The focused
[decode test](tests/test_decode.py) provides allocations and launches for all six
architecture/precision combinations, starting from zero state. It exercises
CUDA graph replay across multiple flushes without a model or dataset.

Mamba2 is specialized to `head_dim=64`, `dstate=128`, and 16 heads per B/C group.
Inputs/output are BF16; `dt`, `A`, `D` and `dt_bias` are FP32. `D` has shape
`[heads, 64]`. GDN is specialized to `key_dim=value_dim=128`. The FP8 non-flush
kernel accepts one to four value heads per key head; the NVFP4 non-flush kernel
requires three or four. Its q/k/v/a/b and output are BF16, while `A_log` and
`dt_bias` are FP32. Gate rows must satisfy the compiled 16-element stride
alignment. All examples use contiguous buffers.

State and ring tensors have a leading physical-slot dimension. Reserve slot zero
and supply active slot IDs in an INT32 `state_indices` tensor. Inputs have a leading
decode-row dimension. The FP16 recurrence anchor is `[slots, heads, rows, 128]`;
here `rows` is 64 for Mamba2 and 128 for GDN.

| Read-cache metadata | Mamba2 | GDN |
| --- | --- | --- |
| FP8 bytes | `[slots, heads, 64, 128]` | `[slots, value_heads, 128, 128]` |
| FP8 FP32 scales | `[slots, heads, 64]` | `[slots, value_heads, 128]` |
| NVFP4 packed bytes | `[slots, heads/2, 128, 64]` | `[slots, value_heads, 128, 64]` |
| NVFP4 E4M3 scale bytes | `[slots, heads/2, 1024]` | `[slots, value_heads, 1024]` |
| NVFP4 `sg_meta` | FP32 `[slots, 3, groups]` | FP32 `[slots, 3, key_heads]` |
| NVFP4 `sg_parity` | INT32 `[slots, groups]` | INT32 `[slots, key_heads]` |

Two E2M1 elements occupy each NVFP4 byte, with the even state column in the low
nibble. Scale bytes use the kernel's chunk layout, not a flat logical block order.
The two first `sg_meta` planes hold alternating group maxima; plane 2 holds the
outer scale used by the current read copy, replicated across groups. Initialize
the active maximum plane and read-copy scales consistently with the initial state.

FP8/NVFP4 non-flush launches take INT32 positions `0..15`; flush launches take
INT64 tags `(checkpoint_event << 4) | position`. FP16 Mamba2 and both FP16 GDN
launches use these INT64 tags. Advance the checkpoint event each window to obtain
new SR bits. Supply INT8 `is_flush` flags, with a flush at position 15. For split
implementations, launch non-flush and flush kernels in order on the same stream;
each skips rows belonging to the other phase. FP32 comparators accept non-flush
positions only and do not form a full low-precision decode loop.

Seeds and layer salts identify the rounding stream; the fixed Philox construction
uses seven rounds. Request-slot reassignment changes the stream.
