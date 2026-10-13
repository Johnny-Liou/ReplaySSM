# ReplaySSM

Code for the paper [*ReplaySSM: Cache SSM Inputs, Not State*](https://arxiv.org/abs/2610.13778)
by Ze-Wei Liou, Harris Nover, Amey Naik, Andrii Skliar, Igor Shovkun, Ido Shahaf
and Tri Dao (Princeton University and NVIDIA).

[arXiv](https://arxiv.org/abs/2610.13778) ·
[NVIDIA Technical Blog](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/) ·
[Dao AI Lab blog](https://dao-lab.ai/blog/2026/replayssm/)

The NVIDIA post summarizes the paper and its deployments, and the earlier Dao AI
Lab post gives a longer walkthrough of standard and speculative decoding.

ReplaySSM runs in production in GLM-5.3-Flash and Kimi-K3 and is integrated in
TensorRT-LLM, vLLM, SGLang and FlashInfer. See [Adoption](#adoption) for details.

ReplaySSM represents an SSM state as a checkpoint and a bounded buffer of recent
inputs. This separates output computation from checkpoint updates, with three
benefits:

- **Reduced memory traffic:** most decoding steps compute outputs without writing
  the state. Periodic flushes update the checkpoint, roughly halving the dominant
  state memory traffic.
- **Cheap rollback:** speculative decoding discards rejected inputs by moving a
  buffer pointer. Output-only verification avoids materializing intermediate
  recurrent states.
- **Separate state precisions:** non-flush steps use a low-precision checkpoint
  for output computation, while flushes use a higher-precision checkpoint for
  recurrent updates. Rounding error in the low-precision copy never enters later
  recurrent updates.

This repository is a vLLM fork with ReplaySSM implementations for Mamba2 and
Gated DeltaNet (GDN), covering standard and speculative decoding, along with
standalone precision kernels. Upstream vLLM has merged the Mamba2 standard
decoding kernel, and its FlashInfer backend covers Mamba2 standard and
speculative decoding. The remaining Triton kernels, the emulated quantization
and the low-precision CuTe kernels are only available here.

| Code | Location |
| --- | --- |
| Mamba2 standard decoding | [Triton kernel](vllm/model_executor/layers/mamba/ops/selective_state_update_replayssm_output_only.py) |
| Mamba2 speculative decoding | [Triton kernel](vllm/model_executor/layers/mamba/ops/selective_state_update_replayssm_spec.py) |
| GDN standard decoding | [Triton kernel](vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py) |
| GDN speculative decoding | [Triton kernel](vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py) |
| Emulated quantization | [emulation/](emulation/README.md): nine precision presets, FP32 storage |
| Low-precision CuTe kernels | [low_precision/](low_precision/README.md): FP16+SR, FP8 and NVFP4 on B300 |

The vLLM integration uses `--use-replayssm` for standard decoding and
`--use-replayssm-spec` alongside a speculative decoding configuration. The flags
are mutually exclusive. `--replayssm-buffer-len 16` selects the paper's window
length. Relevant metadata and dispatch live in
[mamba_attn.py](vllm/v1/attention/backends/mamba_attn.py),
[gdn_attn.py](vllm/v1/attention/backends/gdn_attn.py), and the
[Mamba layers](vllm/model_executor/layers/mamba/).

The vLLM integration is based on upstream commit
`ee1d99636702b8ed9ad88c2c2b833d331dce01c1`, with ReplaySSM extensions.

Build the serving engine from this checkout using the upstream
[source-build instructions](docs/getting_started/installation/gpu.cuda.inc.md#full-build).
Set `VLLM_VERSION_OVERRIDE=0.23.1+replayssm` when installing this snapshot without
upstream Git tags. The original project overview is retained in
[README.vllm.md](README.vllm.md). The two precision folders have separate setup
instructions and do not require a vLLM build.

The [emulation kernels](emulation/README.md) implement separate read and recurrence
precision and expose their quantization behavior. The [CuTe kernels](low_precision/README.md)
support the latency benefit of reading the checkpoint in low precision on B300.
Both folders include focused correctness tests and usage instructions.

## Benchmarks

[e2e_decode_speedup.py](benchmarks/replayssm/e2e_decode_speedup.py) times standard
decoding with the conventional kernel and with ReplaySSM.
[e2e_spec_decode_throughput.py](benchmarks/replayssm/e2e_spec_decode_throughput.py)
compares speculative decoding throughput against standard decoding and vLLM's
native speculative decoding.

```bash
python benchmarks/replayssm/e2e_decode_speedup.py --model-id nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
python benchmarks/replayssm/e2e_decode_speedup.py --model-id Qwen/Qwen3.5-4B
python benchmarks/replayssm/e2e_spec_decode_throughput.py --batch-size 512 \
    --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

## Adoption

ReplaySSM is used in production models. [GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash)
runs it in its serving stack, and the Z.ai
[infrastructure report](https://z.ai/blog/glm-built-its-inference-infrastructure)
describes adding it to the KDA decode kernel.
[Kimi-K3](https://arxiv.org/pdf/2607.24653) (Section 5.4.2) caches the projected
draft inputs for MTP verification in its KDA layers and cites ReplaySSM as a
concurrent approach.

ReplaySSM is also integrated in TensorRT-LLM, vLLM, SGLang and FlashInfer, as
summarized in the
[NVIDIA Technical Blog post](https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/).
The paper authors contributed the merged pull requests below, except the row
marked "NVIDIA", which was contributed independently.

| System | Scope | Link | By |
| --- | --- | --- | --- |
| TensorRT-LLM | Mamba2 MTP, checkpointing replay | [#14203](https://github.com/NVIDIA/TensorRT-LLM/pull/14203) | paper authors |
| TensorRT-LLM | GDN MTP replay | [#16464](https://github.com/NVIDIA/TensorRT-LLM/pull/16464) | NVIDIA |
| vLLM | Mamba2 standard decoding | [#48018](https://github.com/vllm-project/vllm/pull/48018) | paper authors |
| vLLM | FlashInfer backend, Mamba2 standard decoding and MTP | [#52506](https://github.com/vllm-project/vllm/pull/52506), [#52928](https://github.com/vllm-project/vllm/pull/52928) | paper authors |
| FlashInfer | Mamba2 checkpointing, MTP | [#3975](https://github.com/flashinfer-ai/flashinfer/pull/3975) | paper authors |
| FlashInfer | GDN speculative decoding, u/d cache | [#4081](https://github.com/flashinfer-ai/flashinfer/pull/4081) | paper authors |
| FlashInfer | Materialization for prefix caching | [#4815](https://github.com/flashinfer-ai/flashinfer/pull/4815) | paper authors |

Pull requests in progress: FlashInfer
[#4140](https://github.com/flashinfer-ai/flashinfer/pull/4140) and
[#6274](https://github.com/flashinfer-ai/flashinfer/pull/6274);
vLLM
[#48792](https://github.com/vllm-project/vllm/pull/48792),
[#49847](https://github.com/vllm-project/vllm/pull/49847) and
[#49887](https://github.com/vllm-project/vllm/pull/49887)

## Citation

If you use ReplaySSM in your work, please cite:

```bibtex
@misc{liou2026replayssmcachessminputs,
      title={ReplaySSM: Cache SSM Inputs, Not State},
      author={Ze-Wei Liou and Harris Nover and Amey Naik and Andrii Skliar and Igor Shovkun and Ido Shahaf and Tri Dao},
      year={2026},
      eprint={2610.13778},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.13778},
}
```
