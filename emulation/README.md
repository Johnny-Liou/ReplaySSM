# Emulated quantization

Standalone Triton kernels for separate recurrence and read precision in standard
decoding. Quantized values are stored as dequantized FP32 tensors to examine the
effect of checkpoint rounding on the recurrence and output computation.

The recurrence anchor and read copy are formed independently from the same
unrounded FP32 refreshed state. The read copy never becomes the next recurrence
anchor. Checkpoints refresh every 16 tokens. The first decode output uses the
post-prefill checkpoint before its initial quantization.

| Preset | Recurrence anchor | Read copy |
| --- | --- | --- |
| `s32_rs` | FP32 | Shared |
| `f16_sr_rs` | FP16 + SR | Shared |
| `f8_sr_rs` | FP8 + SR | Shared |
| `f4_sr_rs` | NVFP4 + SR | Shared |
| `tp8_hi32` | FP32 | FP8 + RTN |
| `tp8_hi16_sr` | FP16 + SR | FP8 + RTN |
| `tp4_hi32` | FP32 | NVFP4 + RTN |
| `tp4_hi16_sr` | FP16 + SR | NVFP4 + RTN |
| `tp4_hi8_sr` | FP8 + SR | NVFP4 + RTN |

SR means stochastic rounding; RTN means round to nearest, ties to even. Both
architectures use these names. GDN always uses `unf_vf`: non-flush outputs use
cached FP16 updates, while flushes reconstruct from raw BF16 keys/values and
FP32 gate/beta buffers. The raw value ring is required by this flush.

FP16 is unscaled. FP8 E4M3 has one FP32 scale per 128 elements along the trailing
state dimension. NVFP4 E2M1 has block-16 E4M3 scales and one FP32 outer scale per
request/layer state. A second kernel reduces the request-wide maximum before
quantizing both copies. Blocks whose scale rounds to zero decode to zero.

GDN checkpoint reads and flush contractions use BF16 operands with FP32
accumulation. Normalized keys, residuals and weighted flush updates are rounded
to BF16. Mamba2 flush contractions also use BF16 operands and FP32 accumulation.

## Minimal use

Requires PyTorch and Triton on an NVIDIA GPU with BF16 support. Tested with
PyTorch 2.13.0+cu130 and Triton 3.7.1. From the repository root:

```bash
uv venv emulation/.venv --python 3.12
uv pip install --python emulation/.venv/bin/python torch==2.13.0 --torch-backend=cu130
uv pip install --python emulation/.venv/bin/python triton==3.7.1 pytest
emulation/.venv/bin/python -m pytest emulation/tests/test_emulation.py -q
```

Import `Mamba2Replay` or `GDNReplay`; no vLLM installation is needed.

```python
import torch
from emulation import Mamba2Replay

device = "cuda"
state = torch.zeros(2, 4, 64, 128, device=device)  # post-prefill checkpoint
replay = Mamba2Replay(state, num_groups=2, mode="tp4_hi16_sr")
x = torch.randn(2, 4, 64, device=device, dtype=torch.bfloat16)
dt = torch.full((2, 4), -3.0, device=device)
B = torch.randn(2, 2, 128, device=device, dtype=torch.bfloat16)
C = torch.randn_like(B)
A = -torch.ones(4, device=device)
D = torch.ones(4, device=device)
dt_bias = torch.zeros(4, device=device)
for _ in range(48):
    output = replay.step(x, dt, B, C, A, D, dt_bias)
```

For GDN, construct `GDNReplay(state, key_heads, mode)` with state shape
`[batch, value_heads, value_dim, key_dim]`. Call `step(qkv, a, b, A_log, dt_bias)`:
`qkv` is contiguous BF16 `[batch, 2*key_heads*key_dim + value_heads*value_dim]`,
packed as q, k, v. `a` and `b` are FP32 `[batch, value_heads]`; `A_log` and
`dt_bias` are FP32 `[value_heads]`. Queries and keys are L2-normalized internally,
with query scale `key_dim**-0.5`.

The wrappers require contiguous inputs, fixed batch membership, FP32 initial
state and power-of-two state columns of at least 128. Every call advances one
token for every request; device counters support CUDA graph replay. The output
buffer is reused, so clone it to retain a history. Initialize a new wrapper for
a new batch or post-prefill checkpoint. The rounding seed is 1234, indexed by
window and physical element offset.

`quant_tl.py` contains quantizers, `mamba2_replay.py` and `gdn_replay_unf_vf.py`
contain the recurrence kernels, and `api.py` provides allocation and launches.
