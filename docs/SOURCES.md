# Source provenance

Code and checkpoint metadata checked on 2026-10-26. Full pins are in [sources.lock.json](../sources.lock.json)
and [recipes/flash17-exl3-2.8bpw.json](../recipes/flash17-exl3-2.8bpw.json).

| Component | Source |
|---|---|
| Runtime/tooling base | [jayleaton `60b66888a1a3`](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/commit/60b66888a1a379fb38bf85c232b27ec60b007373) |
| Serving changes | [0rand `305659970ae6`](https://github.com/0rand/deepseek-v41-tensorfold-spark/commit/305659970ae63729b0753b27dea85a7ac4b5a93b), `pr65-flash17-serving` |
| Common ancestor | `853310dab967480013eda7cef4e79c2fd041629a` |
| Asset Python reference | [ashhart/TensorFold `c4646171139e`](https://github.com/ashhart/TensorFold/commit/c4646171139ee8a3c38103eaa1699dad226ec12b), plus the base repository's three Python patches |
| Served checkpoint | [jayleaton EXL3 2.8 bpw, revision `859396591c34`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw/tree/859396591c340195526aaf61f94b4ad5323a5e18) |
| Engram source | [DeepSeek original, revision `1bae0e108899`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) |

## Runtime patches

- **0001:** Applies the fork's runtime delta to the newer base: rank-independent disk charges, boot SSD
  intersection, stable session IDs, restore checks, write-through, telemetry, and prefill-workspace checks.
  Retains upstream WebP and streaming fixes; CUDA kernels and numerical operators are unchanged.
- **0002:** Opens the Engram source before the `EngramNotOpen` guard for SSD-restored slots. Carries forward
  the locally verified repair from the previous deployment; neither public head included it.
- **0003:** Fixes a test's reference descriptor to open without `O_DIRECT`. Changing only the wrapper's Boolean
  did not remove that OS flag. Production I/O is unchanged; the corrected host suite passed.

## Build and assets

The upstream-derived Dockerfile targets ARM64/SM121 with Zig 0.17.0, CUDA 13.0.1, and cuBLAS/cuBLASLt 13.6.0.2.
CUDA base tags and other APT packages are not fully pinned, so independent rebuilds are not bit-reproducible.
The resulting runtime image ID is checked on both ranks.

Assets use digest-pinned NGC PyTorch 26.07; preparation checks Triton 3.7.1 and ptxas 13.3.73. The isolated
Hugging Face client is pinned to 1.24.0. Exact pins live in the source lock.

Orchestration is new code, with [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) as the workflow
reference. Source exports retain upstream licenses, notices, and fixtures.

For updates, review the source diff, change the lock/patches, assemble sources, run the
[checks](VALIDATION.md), then build and prepare under the new runtime ID. Do not reuse stale asset receipts.
