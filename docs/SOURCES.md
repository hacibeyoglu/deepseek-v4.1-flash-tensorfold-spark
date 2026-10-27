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
| Experimental EAQ v1 checkpoint | [Mia EXL3 2.9 bpw, revision `45ee5d0fd106`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/tree/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b) |
| Experimental LibertAI checkpoint | [LibertAI EXL3 2.9 bpw, revision `afc7aa0aab98`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/tree/afc7aa0aab98ee735c0f54e0a122019af1ac08f1) |

## Runtime patches

- **0001:** Applies the fork's runtime delta to the newer base: rank-independent disk charges, boot SSD
  intersection, stable session IDs, restore checks, write-through, telemetry, and prefill-workspace checks.
  Retains upstream WebP and streaming fixes; CUDA kernels and numerical operators are unchanged.
- **0002:** Opens the Engram source before the `EngramNotOpen` guard for SSD-restored slots. Carries forward
  the locally verified repair from the previous deployment; neither public head included it.
- **0003:** Fixes a test's reference descriptor to open without `O_DIRECT`. Changing only the wrapper's Boolean
  did not remove that OS flag. Production I/O is unchanged; the corrected host suite passed.
- **0004:** Accepts FP16 router biases and DSpark confidence weights. Reuses direct FP16-to-FP32 widening;
  existing FP32/BF16 behavior and unrelated tensor restrictions remain intact. Includes both-rank loader tests.
- **0005:** Allows AOT preparation to read the selected pack's config and headers through `TF_DSV41_NEEDS_MODEL`.
  New recipes opt in; the original recipe retains the existing width tables. Includes mixed-width planning tests.

- **0006:** Parallelizes independent AOT decode scans and prefill row enumeration with bounded CPU
  workers and per-thread variant/coverage state. Results merge deterministically; failures propagate after
  joining workers. Adaptive scans retain their original boundaries and numerical kernels are unchanged.
  The head distributes shared tables and verified cubins; the worker validates their hashes and still
  packs/checks rank-specific Engram data. AOT enumeration consumes the shared serving settings and TP
  world size, not TP rank; the existing model-derived width plan already uses rank 0 on both nodes.
  RoPE and Engram host lookup generators take only the shared model config/tokenizer.

The new recipes were checked against code in
[MiaAI-Lab's stack](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/tree/27741e176a9e5b98c10c9c7d012fd3e4fcdc19b6)
and [0rand's Python stack](https://github.com/0rand/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/tree/887bec7bc0a45286d0d3ae7a2ed04fd067fb9d55).
Their engines and display-KV changes are not transplanted. All 39 EAQ and 43 LibertAI shard headers were inspected;
each pack has 192,495 tensors. EAQ v1.1's mixed BF16 linears require separate runtime work.

`model.verification` selects the checksum source, defaulting to `sha256sums`; the new recipes select `hub-sha256`.
Downloads check pinned revisions, file sizes, small Git-file blob hashes, and index coverage, and record expected
tensor hashes. Tensor checksum scans require `verify-weights` or `--verify-weights`. Worker checks stay offline;
cached checkpoint files stay unchanged.

## Build and assets

The upstream-derived Dockerfile targets ARM64/SM121 with Zig 0.17.0, CUDA 13.0.1, and cuBLAS/cuBLASLt 13.6.0.2.
CUDA base tags and other APT packages are not fully pinned, so independent rebuilds are not bit-reproducible.
The resulting runtime image ID is checked on both ranks.

Assets use digest-pinned NGC PyTorch 26.07; preparation checks Triton 3.7.1 and ptxas 13.3.73. The isolated
Hugging Face client is pinned to 1.24.0. Exact pins live in the source lock.

Orchestration is new code, with [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) as the workflow
reference. Source exports retain upstream licenses, notices, and fixtures.

Transfer behavior reviewed on 2026-10-26 against upstream
[`hf-download.sh`](https://github.com/eugr/spark-vllm-docker/blob/main/hf-download.sh),
[`build-and-copy.sh`](https://github.com/eugr/spark-vllm-docker/blob/main/build-and-copy.sh), and
[`autodiscover.sh`](https://github.com/eugr/spark-vllm-docker/blob/main/autodiscover.sh): head downloads followed
by cache rsync, and Docker save/load with matching-image skips. This stack uses the same approach over a
verified direct fabric SSH destination. It limits cache copying to pinned snapshots and their required blobs,
checks worker files offline, and also distributes the asset-preparation image from the head. Tensor hashes and
rsync checksum scans are now opt-in; default transfers reuse immutable cached blobs by size.

For updates, review the source diff, change the lock/patches, assemble sources, run the
[checks](VALIDATION.md), then build and prepare under the new runtime ID. Do not reuse stale asset receipts.
