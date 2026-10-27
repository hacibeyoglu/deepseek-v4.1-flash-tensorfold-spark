# Recipe settings and their origins

Reviewed 2026-10-26 against the pinned launchers and engine readers linked below. These are translations
into this repository's combined Zig runtime, not copies of another engine's environment. The experimental
2.9 bpw profiles still require ARM64/GB10 preparation and GPU qualification.

## Effective profiles

| Setting | Jayleaton 2.8 bpw | Mia EAQ v1 2.9 bpw | LibertAI 2.9 bpw |
|---|---|---|---|
| Context per request | 1,048,576 | 1,048,576 | 1,048,576 |
| Concurrent slots | 4 | 4 | 8 |
| Shared KV pool positions | 2,100,000 | 1,048,576 | 1,048,576 |
| KV split / compact exchange | On / on | Off / off | Off / off |
| Expert routing | Top-p 0.85 | Full configured top-k | Full configured top-k |
| Prefill chunk / rows | 4096 / 4096 | 2048 / 2048 | 2048 / 2048 |
| 4K prefill opt-in | On | Off | Off |
| Decode/verify row cap | 64 | 16 | 16 |
| Sampling defaults: temperature / top-p / top-k | Checkpoint/runtime defaults | 1.0 / 0.95 / 20 | 0.4 / 1.0 / 0 |
| DSpark depth policy | Existing calibrated cost policy | Cost policy, cap 5 | Cost policy, cap 5 |
| SSD sessions per rank | 128 GiB | Off; RAM reuse retained | 32 GiB |
| Minimum tokens for SSD entry | 1024 | Not applicable | 4096 |

`serve.slots` becomes `TF_DSV41_SLOTS` on both ranks and `--parallel` on the head. The runtime's
[slot reader](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/60b66888a1a379fb38bf85c232b27ec60b007373/engine/zig/zig/src/families/deepseek_v41/slots.zig)
accepts 1–16; this launcher exposes 1–8 to cover the selected source profiles. Supporting eight in command
generation does not establish eight-request memory fit or throughput.

Context limits need not be multiples of 256. Runtime patch 0007 makes AOT planning round only the internal
page tables, as the serving allocator already does; a limit such as 500,000 remains exactly 500,000 tokens.
The shared pool is a separate capacity and is not resized to the request limit.

Sampling is stored as optional `serve.temperature`, `serve.top_p`, and `serve.top_k` and passed as native
CLI flags, because this runtime has no corresponding `TF_DSV41_*` environment readers. A top-k of zero
disables that sampling cut. Requests can override these defaults. Sampling-only changes require recreation;
they do not change kernel assets. Slot and environment changes select fresh asset/session namespaces.

## Source mapping

### Jayleaton

Source: [flash17 environment](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/60b66888a1a379fb38bf85c232b27ec60b007373/config/prod-zig-flash17.env.example)
at `60b66888a1a379fb38bf85c232b27ec60b007373`, the runtime/tooling base in `sources.lock.json`.
Its inference environment is retained. SSD capacity and minimum entry length now match its 128 GiB / 1024
values instead of the local 32 GiB / zero override. The source launcher supplies no sampling overrides;
the baseline continues to inherit checkpoint/runtime defaults.

### Mia EAQ v1

Source: [settings](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/27741e176a9e5b98c10c9c7d012fd3e4fcdc19b6/scripts/config.sh),
[launcher](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/27741e176a9e5b98c10c9c7d012fd3e4fcdc19b6/start.sh), and
[spill/sharding defaults](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/27741e176a9e5b98c10c9c7d012fd3e4fcdc19b6/scripts/port.sh)
at `27741e176a9e5b98c10c9c7d012fd3e4fcdc19b6`.

- `CONTEXT=1048576`, `PARALLEL=4`, no explicit `POOL`, and `KV_SHARD=0` map to four slots with a
  1,048,576-position replicated pool. The pool is shared, not one full window per concurrent request.
- `TEMPERATURE=1.0`, `TOP_P=0.95`, and `TOP_K=20` become explicit native serve flags.
- The pinned `scripts/port.sh` defaults `PORT_SPILL_DEFAULT_GIB=0`, even though parts of its README still
  discuss 64 GiB. Match executable defaults: an empty `TF_DSV41_SESSION_DISK` disables this runtime's disk tier.
- Its pinned [model](https://github.com/bertholomus/TensorFold/blob/629c524a0ea57512a5342abb498d1b7c25353343/zig/src/families/dsv41/model.zig)
  uses 2048-row chunks; the source launch uses 16-row rounds. Set both local prefill sizes to 2048,
  disable `TF_DSV41_PF_4K`, remove its memory opt-in, and set `TF_DSV41_ROWS_CAP=16`.
- The source routes the configured experts without Jayleaton's optional expert top-p cut. Omit
  `TF_DSV41_EXPERT_TOPP`, which disables that cut in this runtime.

The [EAQ v1 card](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/blob/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b/README.md)
describes standard EXL3 compatibility, not a requirement to use the 2.8 bpw serving profile.

### LibertAI

Source: [environment](https://github.com/0rand/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/887bec7bc0a45286d0d3ae7a2ed04fd067fb9d55/.env.example)
and [launcher](https://github.com/0rand/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/887bec7bc0a45286d0d3ae7a2ed04fd067fb9d55/start.sh)
at `887bec7bc0a45286d0d3ae7a2ed04fd067fb9d55`. This is the ordinary Python serving lane;
display-KV settings are separate experiments and are not copied.

- `PARALLEL=8`, `PREFILL_CHUNK=2048`, `SPILL_GIB=32`, `TEMPERATURE=0.4`, `TOP_P=1.0`, thinking,
  high effort, vision, and DSpark are reflected in the local recipe.
- The pinned checkpoint contains no `generation_config.json`, and this Python stack supplies no top-k cut;
  local `serve.top_k=0` makes that explicit.
- The source [MoE path](https://github.com/Elevatormusic/TensorFold/blob/07aa02585f64cfb04c60a96de6554c0d4708ed46/src/tensorfold/families/deepseek_v41/cuda/model.py)
  uses its configured top-k without an expert top-p cut. The shared cache is replicated and spans the context.
- Its [spill reader](https://github.com/Elevatormusic/TensorFold/blob/07aa02585f64cfb04c60a96de6554c0d4708ed46/src/tensorfold/families/deepseek_v41/cuda/spill.py)
  defaults `TF_DS_SPILL_MIN` to 4096; map this to `TF_DSV41_SESSION_DISK_MIN=4096`.

## Differences that remain

- **Precision:** checkpoint pins and FP8 index KV remain unchanged. Native compressed main KV and index KV
  are different structures; the EAQ card's native-FP4 guidance is not an instruction to change index KV.
- **Memory:** retain the local 5 / 4 GiB admission/hard floors, measured boot check and 3.5 GiB workspace
  floor. The source Python allocator's `MEM_FLOOR_GIB=1.0` and Mia's external memguard are different policies;
  neither is a direct substitute for these native admission checks.
- **Drafting:** `MTP_DRAFTS=5` maps to `TF_DSV41_DRAFT_DEPTH=5` with `TF_DSV41_DEPTH=cost` for both new
  recipes. The source engines' confidence policy, per-concurrency depth table and round-cost tables are not
  identical to this runtime's calibrated cost policy. This is a cap match, not identical draft scheduling.
- **Transport:** keep `GLM53_TF_ROCE_MAX_KB=1024`. The source engines' 4.25 MiB rings cannot be copied:
  the pinned native transport rejects RDMA slots larger than 1 MiB. It handles larger transfers differently.
- **Scheduling and kernels:** keep this runtime's supported graphs, prefetch, replay, grammar and native-image
  paths. Source `TF_DS_*` flags for prefill interleave timing, eager absorption, page-cache advice, allocator
  limits, RDMA idle naps and display-KV have no exact recipe-only equivalents. Do not inject ignored variables.
- **Sessions:** retain this runtime's RAM-store implementation and immediate write-through where disk is enabled;
  matching disk budget/minimum size does not reproduce another engine's eviction, retention or file format.

No model or KV precision, checkpoint pin, runtime source lock, or runtime patch was changed for this translation.
Use `./stack prepare --recipe NAME` and `./stack start --recipe NAME` with the same profile after stopping the
stack. Existing generated assets and session files are retained; changed profiles use new namespaces.
