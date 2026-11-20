# Source provenance

Current source layout, reviewed 2026-11-19.

Native serving builds upstream
[`ashhart/TensorFold@f212eeabd`](https://github.com/ashhart/TensorFold/commit/f212eeabde0fc2ac3234ca947a69cc7714abaca2)
and applies the reviewed series in [patches/engine](../patches/engine), `0001`-`0091`, exactly as
upstream's recipe does: a Git checkout at the pin, `git am` with a fixed committer, then checks. Each
patch's SHA-256 must match the lock; the resulting Git tree must equal
`d73f715ca9892d6f8801f33d2164780390490746` (upstream's `config/tensorfold.tree`); and the assembled
content must match the lock's `tree`. The series reproduces upstream production engine `6c4eeccda`
(the G19 line) except for 24 comment lines naming internal machines, tools and people.

The recipe repository at [`24a84eb`](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/commit/24a84eb)
supplies asset-generator references only; its `engine/`, `scripts/pack_engram.py` and
`config/pfdense-table.json` are unchanged at upstream `13db91e`. Its deleted Python patches come from
the separately pinned, hash-checked `python_patches` entry. The seven older runtime patches below
remain archived under `legacy_patches`; `patches` (asset-source patches) is empty.

## Engine migration, 2026-11-19

Migrated the native engine from the jayleaton candidate branch (`12257eb51fd` plus `0090`/`0091`)
to upstream TensorFold `f212eeabd` plus the 91-patch series published with upstream recipe `13db91e`.
After a fetch the previous engine is retained at `.stack/engine.previous`; rollback is a lock revert
plus a fetch and rebuild, not automatic.

Why: the old pin was upstream's *previous* recipe engine. The new base is 2,517 upstream commits
newer and carries the shared compute lane and runner, engine-owned reply bytes across prose strips,
`TF_DSV41_ENGRAM_READERS`, `TF_DSV41_PROMPT_PROF`, tool-call ids keyed over the request, chunked
prefill admission, prompt-cache retention and the Engram direct-read fix. The series adds:

- `0073`-`0087`: the Qwen recipe's lane and runner fixes (resumed segments, budget holds across tool
  results, digit stops, quoted inline abstention, stop receipts, lean `stream.zig`).
- `0088`: DSpark staging regions padded to 8 bytes, so a 16-slot pass above 12 live requests no
  longer panics. This launcher still exposes at most eight slots.
- `0089`: optional load-adaptive context window (`TF_DSV41_CONTEXT_BUDGET`, `_FLOOR`, `_HOLD_S`),
  off unless set.
- `0090`, `0091`: unchanged from the previous review.

Not adopted: the CUDA 13.3 forward-compatibility layer upstream layers over its runner image. This
image keeps its CUDA 13.0.1 base and pinned cuBLAS 13.6.0.2.

The new engine no longer reads `TF_DSV41_RUNNER_SPEC_HOLD`; the launcher refuses it with the
replacement note in `lib/config.py`. Every non-launcher setting in the shipped recipes is read by the
new engine (checked against its 710-name manifest).

## Upstream review, 2026-11-16 (historical)

Reviewed upstream `13db91e` (2026-11-16), 14 commits after the previously pinned `24a84eb`
(#89-#103). The retained `engine/` asset-generator tree and the `python_patches` pin are unchanged.

Adopted:

- The engine pin moves from `7b79ef92d846` to `12257eb51fd` on
  `dsv41/undeclared-tool-passthrough`, four commits: #80's prompt checkpoints and checkpoint-spacing
  fix, and the two undeclared-tool commits. The engine logs costly session misses;
  `TF_DSV41_SESSION_LCP_RESUME` stays off by default, and no recipe enables it. A whole tool call to an
  undeclared name is returned as a `tool_calls` entry instead of text, matching vLLM. Upstream's
  OpenCode 2.2.7 report (four failed trials of 30) and its timings are upstream evidence, not
  measurements on these Sparks.
- Engine patches `0090` and `0091` are applied after export, with SHA-256 and assembled-tree checks.
  `0090` makes identical short prompts repeat under `TF_DSV41_PREFILL=replay` (our recipes use replay;
  a prompt of 17 tokens or fewer could differ run to run). `0091` drops an armed or pending trace when
  the think block closes under `TF_DSV41_RUNNER_SPANS=reasoning`; the shipped compute recipe leaves the
  span setting unset, so its current all-span behavior is unchanged.
- The source lock recorded the previous pin as `fallback_commit`. The 2026-11-19 migration removes
  that field; rollback is still a manual lock revert, fetch and rebuild.

Not adopted:

- Patch `0088` (DSpark staging alignment) was not vendored then. The 2026-11-19 migration includes it;
  it fixes a panic above 12 live slots, while four and eight slots are unchanged upstream.
- Upstream #91/#92/#94/#98 rebase the engine on upstream `f212eeabd` plus G's 72-patch series and the
  Qwen compute-lane patches `0073`-`0087`, add adaptive context (`0089`), and move the image to a CUDA
  13.3 forward-compatibility layer. That is a 2,517-commit upstream rebase with new AOT and prepared
  assets and a different image build; it needs a separate TP=2 and live qualification. The eight-slot
  default/192K option and adaptive context are not adopted.
- Upstream's watchdog, thermal policy, authentication, site configuration and `results/` evidence are
  not vendored. The undeclared-tool rows are referenced, not copied.

The engine bump changes the runtime fingerprint. Prepared weight images are reused only when their
native format key matches; AOT assets must be regenerated for the new engine tree. Build, prepare and
start in a maintenance window; see [validation](VALIDATION.md#upstream-review-2026-11-16).

## Previous upstream adoption, 2026-11-09

Compared upstream `ed96ef610693` with `24a84eb`: two commits, #87 and #88. The retained `engine/`
asset-generator tree is byte-identical across that comparison.

- #87 replaces candidate G with the compute-lane release, adds `deadline_s`, the experimental in-model
  runner, thinking-only default-tool admission, preservation of a client's own `python` tool, and
  accounting of executed compute. The optional compute recipe follows upstream's nineteen compute
  settings and leaves `TF_CUDA_SCHED` unset, as its measured profile does. Existing recipes keep their
  prior settings, including `TF_CUDA_SCHED=blocking`, and explicitly disable the four compute switches.
- #88 includes upstream TensorFold #2182: the NVMe session reader accepts replay stashes longer than
  the ring and valid drafter offsets without drafter rings, fixing the reported `BadBlob` restore failure.
  This is a source fix; upstream's timings are not measurements on Emre's Sparks.
- The shared image builds `tf-runner-frontend-check`, installs clang/lld 18 and includes the pinned
  runner circuit and graph bank. Both the release archive and every extracted file are SHA-256 checked.
  The image build and image check require the frontend self-test. Only the head downloads/builds;
  workers receive the image through the existing verified fabric transfer.
- `scripts/fetch_runner_assets.py`, `scripts/check_runner_assets.py`, the release pin and file manifest
  are adopted from upstream `24a84eb`. The Python helpers use ordinary CLI output here instead of the
  upstream launcher's `safe_output` adapter; integrity and extraction checks are retained. They run
  through Docker during `./stack build`, not on the worker or during ordinary setup discovery.
- The runtime fingerprint includes runner manifests and helper bytes. Runtime host tests now copy
  `.stack/engine`, the actual serving source, instead of the retained legacy generator engine.

Model revisions, EXL3 bit rates, FP8 index KV and existing capacities are preserved. No benchmark archive was vendored into tracked files, and no upstream watchdog, thermal policy,
authentication change or site configuration was adopted. For optional compute
behavior, evidence limits and commands, see [COMPUTE.md](COMPUTE.md).

The dated reviews and runtime identities below describe earlier export-based builds, not this native
engine. This source adoption did not build or deploy a serving image.

Code and checkpoint metadata checked on 2026-10-26; `main-0rand` runtime equivalence rechecked on
2026-10-27; the runtime/tooling base moved to upstream `391d677b` on 2026-10-30 and to `ed96ef610693`
on 2026-11-03. That bump was reverted on 2026-11-04 after a serving slowdown and re-applied the same
day for a re-test; the slowdown is still unattributed (details below). Full pins are in
[sources.lock.json](../sources.lock.json) and [recipes/flash17-jay-exl3-2.8bpw.json](../recipes/flash17-jay-exl3-2.8bpw.json).
Serving defaults, source-to-runtime mappings and deliberate differences are in [recipe settings](RECIPE-SETTINGS.md).

| Component | Source |
|---|---|
| Native engine | [ashhart `f212eeabd`](https://github.com/ashhart/TensorFold/commit/f212eeabde0fc2ac3234ca947a69cc7714abaca2) |
| Engine patches | [upstream `13db91e`](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/commit/13db91e) `patches/0001`-`0091`, SHA-256 and both trees pinned in the lock |
| Asset-generator reference | [jayleaton `24a84eb`](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/commit/24a84eb) |
| Serving changes | [0rand `305659970ae6`](https://github.com/0rand/deepseek-v41-tensorfold-spark/commit/305659970ae63729b0753b27dea85a7ac4b5a93b), `pr65-flash17-serving` |
| Common ancestor | `853310dab967480013eda7cef4e79c2fd041629a` |
| Asset Python reference | [ashhart/TensorFold `c4646171139e`](https://github.com/ashhart/TensorFold/commit/c4646171139ee8a3c38103eaa1699dad226ec12b), plus the base repository's three Python patches |
| Python patches | [jayleaton `391d677b300d`](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/commit/391d677b300d9b53e64f4265a261c0b329ea58e0) `patches/`, SHA-256 pinned in the lock |
| Served checkpoint | [jayleaton EXL3 2.8 bpw, revision `859396591c34`](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw/tree/859396591c340195526aaf61f94b4ad5323a5e18) |
| Engram source | [DeepSeek original, revision `1bae0e108899`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) |
| Experimental EAQ v1 checkpoint | [Mia EXL3 2.9 bpw, revision `45ee5d0fd106`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/tree/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b) |
| Experimental LibertAI checkpoint | [LibertAI EXL3 2.9 bpw, revision `afc7aa0aab98`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/tree/afc7aa0aab98ee735c0f54e0a122019af1ac08f1) |

## Upstream update, 2026-11-03

Moved the runtime/tooling pin from `391d677b30` to upstream `main` `ed96ef610693` (2026-11-03), five
commits later. The vendored engine is still TensorFold 1.0.2 (`f8fe17d2`) plus the DeepSeek-V4.1 export
`9131e10a`, and this stack still builds the `engine/zig` source export rather than the TensorFold pin
that upstream's own `docker/zig/Dockerfile` now fetches from `config/tensorfold.rev`.

- `44b3dcd` (#81) removed `patches/` from the recipe tree and switched upstream's build to a TensorFold
  pin. `fetch()` globbed that directory, and `Path.glob` on a missing directory returns no matches, so
  the three Python patches would have been skipped while a success manifest was still written. They are
  now read from the `python_patches` lock entry pinned to `391d677b300d`, which still carries them
  byte-identically, and each file's SHA-256 is checked before it is applied.
- `d3ee509` (#82) hashes an image's runtime config into the pair identity and takes guard limits from
  `/etc/spark/thermal.env`; `dcbc795` (#83) pins TensorFold release candidate G; `ed96ef6` (#85) turns
  the compute lane on in the flash17 profile. All three describe upstream's own launcher and fork pin,
  which this stack does not import.
- `33cb5e8` (#78) bounds index position and scratch allocations for intermediate prompt positions and
  ragged segments during 4K prefill. This is the only engine-code change that affects this stack; it is
  included in the assembled export.
- `391d677` (#71) adds the optional STREAM_RB=4 Triton path. The recipe does not set
  `TF_DSV41_STREAM_RB`, so serving behavior is unchanged; enabling it would require regenerated AOT
  variants and separate TP=2 validation.

The export was assembled with `./stack fetch`; patches **0001–0007** applied with `--fuzz=0` (two files
kept GNU patch `.orig` backups after hunks applied at offsets). Upstream's new `planIndexScratch` and
patch **0001**'s workspace changes coexist in `forward_prefill.zig`. Runtime identity became
`87425ce1f41f63c9`, so the new pin gets its own asset namespace. Local orchestration suite: 100 tests
passed, shell syntax and the isolated baseline dry-run passed. Runtime host tests, image builds,
downloads and GPU runs were not performed for this update.

The export reads the same 335 `TF_*` / `TENSORFOLD_*` names at both pins, so the recipes are unchanged.
Upstream's `config/tensorfold-settings.txt` and `scripts/engine_settings.py` describe the TensorFold pin,
not this export: their rename map (`GLM53_TF_ROCE_FAST` to `TF_TP_ROCE_FAST`, `TF_DSV41_PLAN_*` to
`TF_TP_PLAN_*`, and others) and the `TF_DSV41_PF_WS_FLOOR_GIB` move to 4.75 do not apply here.

### Revert and re-test, 2026-11-04

Live serving on this pin was much slower than the previous state, so the bump was reverted to
`60b66888a1a3` and then re-applied the same day for a re-test. The slowdown is **not attributed**.

- The leading suspect is #78's `planIndexScratch`: it adds `ix.pos*` roles to the buffer plan and
  enlarges the index roles it finds, and a larger planned workspace can push a prompt piece past
  `TF_DSV41_PF_WS_FLOOR_GIB` and fall back from 4K to 2K prefill segments. That is a reading of the
  diff, not a measurement.
- #71's `STREAM_RB=4` path should be inert, because no recipe sets `TF_DSV41_STREAM_RB` (default `0`).
- Reverting also dropped the `python_patches` reader. It is restored here because upstream deleted
  `patches/` in `44b3dcd`; without it `fetch` silently applies zero Python patches.

Runtime identity is `87425ce1f41f63c9`, so this pin needs its own image build and asset namespace.
Prepared weight images under `.stack/prepared/<model-revision>/` are reused. See
[validation](VALIDATION.md) for what to compare against the reverted baseline.

## Upstream update, 2026-10-30

Moved the runtime/tooling pin from jayleaton `60b66888a1a3` to upstream `main` `391d677b30` (2026-10-30),
seven commits and 107 paths later. The vendored engine is still TensorFold 1.0.2 (`f8fe17d2`) plus the
DeepSeek-V4.1 export `9131e10a`.

- `33cb5e8` (#78) bounds index position and scratch allocations for intermediate prompt positions and
  ragged segments during 4K prefill, with an 88-case regression. This is the only engine-code change that
  affects this stack; it is included in the assembled export.
- `391d677` (#71) adds the optional STREAM_RB=4 Triton path and its FP64 score gate. The recipe does not
  set `TF_DSV41_STREAM_RB`, so serving behavior is unchanged; enabling it would require regenerated AOT
  variants and separate TP=2 validation.
- `fc9da83`, `3968f6b`, `649f73c` and the merge commits add upstream's guard/watchdog subsystem,
  systemd units and documentation. This stack does not import that orchestration layer: `./stack`
  remains the lifecycle, the engine still serves on `PORT=8000`, `TF_DSV41_PLAN_PIN` stays `auto`,
  `TF_DSV41_PF_WS_FLOOR_GIB` stays 3.5 GiB, and no host CPU set or `TF_CUDA_SCHED` is imposed.

The export was assembled with `./stack fetch`; patches **0001–0007** applied with `--fuzz=0` (two files kept
GNU patch `.orig` backups after hunks applied at offsets). Repeat fetch reported the pinned
exports already match. Runtime host tests were not run for this update: the development host has no Docker,
and the live head had only about 5 GiB available. Local orchestration suite: 97 tests passed, shell syntax
and the isolated baseline dry-run passed. Runtime identity became `8938cc1d2b7c56a7`; assembled source hash
`8bfddddb470a5355154c43005580e30f243523d78d3e8f5ae26ccada0cfec6cf`.

## main-0rand review, 2026-10-27

Reviewed [0rand `055c4aefa879`](https://github.com/0rand/deepseek-v41-tensorfold-spark/commit/055c4aefa8794f1876e2894d666693f3784bb0fa),
the `main-0rand` head at review time. Exporting upstream `60b66888a1a3`, applying patch **0001**
with `--fuzz=0`, and comparing file paths and bytes against that branch found **all 1,801 files
under `engine/zig` identical**, with no missing or extra files. Its upstream merge therefore adds
no runtime changes beyond this stack's existing combination. WebP decoding and streamed Anthropic
prompt-token accounting are already included. Local patches **0002–0007** remain necessary additions;
switching directly to the branch would discard them.

The new `scripts/zig-stack-slots8.sh` is a site-specific wrapper requiring an untracked
`config/prod-zig-slots8.env` and a fixed image tag. It is not a portable eight-slot recipe or evidence
that eight slots fit this deployment. Keep the four-slot profile; any capacity change requires separate
preparation and memory/concurrency qualification. Upstream launchers are reference material; this stack
uses its own head-managed `./stack` lifecycle.

Updated the README's branch link. The source lock intentionally retains the original serving-delta
commit: changing provenance-only lock fields would change the runtime fingerprint and invalidate
matching assets without changing runtime code. No build, preparation, service restart, precision change,
or live benchmark was needed for this review.
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
- **0007:** Makes AOT pool planning use the serving allocator's ceiling page count for a request's context,
  and the separately rounded configured pool for physical tensor sizes. Fixes `exact division produced remainder`
  for contexts such as 500,000 without changing the request limit or pool budget. Host regressions compare
  replicated and split plans against the real allocator on both ranks, including odd slot-page counts.

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

For staged commands, updates and cache handling, see the [build and preparation guide](BUILD.md).

The upstream-derived Dockerfile targets ARM64/SM121 with Zig 0.17.0, CUDA 13.0.1, and cuBLAS/cuBLASLt 13.6.0.2.
CUDA base tags and other APT packages are not fully pinned, so independent rebuilds are not bit-reproducible.
All recipes share `spark-ds41-tensorfold:latest`. Runtime builds retain only that stack runtime tag.
The source-lock/patch/Dockerfile fingerprint is checked as a label to reuse matching images or update these
same tag when inputs change. It still keys prepared assets. The resulting runtime image ID is checked on
both ranks, and launches pin that ID. Host tests build a temporary compiler image and remove it in a
cleanup block, including after errors or cancellation. Explicit `--builder-image` overrides remain owned by
the caller. Docker build cache is retained; no global pruning is performed.

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
