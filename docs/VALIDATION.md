# Validation (2026-10-27)

## Prepared-weight reuse and startup (2026-10-27)

The active baseline is `jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw` at
`859396591c340195526aaf61f94b4ad5323a5e18`, native Zig TP2 runtime `efbc1a1b9c78642d`, image
`sha256:f1cc36f6556d26e8f381bc53731924198715d1ef97427d896c4b9e892b8078ce` on both nodes.
Four slots, 1,048,576 context, the split 2,100,224-token shared pool, model/KV precision, DSpark,
native vision settings and memory guards were retained. No runtime/image rebuild or asset regeneration
was needed for this orchestration change.

| Stage | First conversion start | Cached start |
|---|---:|---:|
| Head runtime load to API serving | 362.6 s | 120.0 s |
| Head weight load | 337.6 s | 109.4 s |
| Head prepared files read / built / written | 0 / 44 / 44 | 44 / 0 / 0 |
| Worker cached weight load | — | 109.2 s |

These are **different cache states**, not an A/B speedup from changing an inference setting. The first
boot created native weight images; the later boot reused them. On the cached boot each rank spent about
98 seconds reading 95.1 GiB of images; remaining startup work included uploads and serving initialization.
The measured whole `stack start` command took 130.1 seconds, including preflight, polling and generation.
Disk/page-cache state was not forced or flushed, and these are single observations rather than repeated
benchmarks. No other inference workload was active.

The launcher now retains `.stack/prepared/<model-revision>/` across runtime updates. Native format,
builder-source, shard metadata, TP and DSpark keys still determine compatibility. Preflight adopted all
44 completed files per rank from the old runtime-specific namespace by hard link, with no payload copy
or checksum scan. Old files/paths remain available; incomplete `.tmp` files are ignored. This avoids
unnecessary conversion after unrelated runtime changes, but cannot avoid conversion when the native
weight-image key changes. Matching AOT assets are still required after a runtime update.

A controlled restart also exposed a pre-existing false occupied-port error from the stopped API's
`TIME_WAIT` sockets. The preflight bind probe now sets `SO_REUSEADDR`, accepting these closed connections
while still rejecting live listeners on Linux. No unrelated process was stopped.

- `./scripts/test.sh`: 97 tests, shell syntax checks and isolated two-node command generation passed.
  New regressions cover real filesystem cache adoption, no payload reads/copies, existing-file preservation,
  rank/revision separation, ignored partial files/symlinks, and real TCP listener/closed-connection states.
  The suite needed loopback permission on the client; the TCP test also passed on spark01 ARM64.
- Both ranks use the intended image and stable prepared mounts, with matching effective commands/env,
  zero restarts and no OOM. API model identity and health passed; the arithmetic canary returned `391`
  with 18 prompt tokens, two completion tokens and no cached prompt tokens.
- The boot guard reported worst serving headroom of 4.89 / 6.41 GiB. Its existing prefill guard retained
  the 4K-to-2K fallback; context/concurrency capacity and generation performance were not requalified.

Evidence and original launcher backups are in `.stack/recovery/startup-cache-20261027/` on each node:
`cold-r0.log` / `cold-r1.log`, `warm-r0.log` / `warm-r1.log`; the head additionally has `cold-launch.json`,
`restart.log` (the port-check failure), `warm-start.log` and `warm-timing.json`.
Use `./stack start --recipe recipes/flash17-jay-exl3-2.8bpw.json` for later launches.

## LibertAI reduced-context startup (2026-10-26)

The live native profile now uses 262,144 context, four slots and a split/compact 524,288-position shared
KV pool. The recipe retains 2K prefill, unchanged model/KV precision, native vision, DSpark and the 5 / 4 GiB
memory floors. Small allocations use the plain arena; expert vectors are reused, stream scratch is lazy,
index workspace is 32 MiB and SSD staging is 16 MiB. See [settings](RECIPE-SETTINGS.md) for the source deviations.

- Both ranks run runtime `efbc1a1b9c78642d`, image
  `sha256:c46b4110391485187f41f56ca225be3c684bc16dcf39976f1e0b7aee2777536e`, with asset namespace
  `8fc8222ebaf694e9`; effective commands/settings match, with zero restarts and no OOM.
- Startup's measured worst serving headroom was **4.34 GiB head / 5.33 GiB worker**, clearing the 4 GiB floor.
  The prior replicated pool failed on the worker at 3.58 GiB; split-only and packed-arena trials also failed.
- Head AOT coverage, copied worker hashes and both rank-specific Engram checks passed.
- API model identity, health and generation passed (`17 * 23` produced `391`). Four simultaneous fresh
  prompts of 12,037–12,038 tokens each returned all four expected arithmetic answers, with zero cached tokens.
  Logs confirm lazy scratch allocation on both ranks. Post-check available memory was about 6.5 / 7.7 GiB.

This validates startup and that text workload, not generation at 262K context, sustained saturation, image/tool
behavior or comparative performance. The local recipe was updated on both nodes; the source recipe's original
settings and all failed-start logs were preserved. Head evidence is under
`.stack/recovery/pool-memory-wnr2yg4m/` (`verified-256k.json`, `concurrency-256k.json`, startup/preparation logs).

Startup-failure handling now reads both owned ranks' log tails before cleanup and preserves the original
error if diagnostics fail. Tests cover both-rank logging before stop and cleanup despite log errors.
`./scripts/test.sh` passed all 92 tests, shell syntax checks and the isolated dry-run.

## AOT planning with partial context pages (2026-10-26)

Runtime `efbc1a1b9c78642d` adds patch 0007. The reported LibertAI setup used a locally edited 500,000-token
context and four slots, with a separate 1,048,576-position replicated pool. `aot-needs` aborted in
`poolFromEnv` because it required the context to divide exactly into 256-token pages. The serving allocator
already rounds the internal slot table up while retaining the exact request capacity.

The planner now uses that allocator's page rounding and the configured physical pool's independently rounded
size. Regressions compare its slot table, compressed/index tensor page counts and split flag with real
allocator objects on both ranks: replicated/split, 2049/500000/500225/1048576 context, and 1,048,576/2,100,000
pool requests. They also cover forced pools, odd slot-page counts and invalid shapes. No context, pool,
precision or numerical kernel is changed by this fix.

- Local orchestration suite: 81 tests passed, shell syntax and isolated dry-run passed.
- ARM64 combined `zig build test test-golden`: all 103 steps passed; the final run executed 259 passing
  tests and skipped five, with the remaining results reused from the same disposable build cache.
- Native ReleaseSafe host executable build: 30/30 steps passed, without CUDA fatbins or GPU execution.
- The built `tf-dsv41-m1 aot-needs` completed against the actual cached LibertAI checkpoint headers with
  the user's unchanged 500,000-token context, four slots and 1,048,576-position pool: 192,215 programs,
  136 kernel variants. It enumerated prefill, images, CED replay and DSpark requirements without a panic.
  This was CPU metadata planning, not Triton compilation, coverage checking or inference.
- Fresh offline source assembly applied the full patch set without fuzz; repeat fetch verified.
  The assembled engine tree matches the tested tree (`8d0f2d836652ef67edbdc9c574775ccf0d398bc34270295c26276bd7d75e25f8`).

Native tests used the existing Zig 0.17.0 builder on spark01 in a disposable source copy, with networking
and GPU access disabled. Both nodes had no running containers before inspection; the user-edited recipe,
managed source exports and model caches were preserved. Raw remote evidence:
`/tmp/tf-aot-pages.ua6URmhr/host-tests-final.log`, `host-build.log`, `real-aot.log` and `fetch-final.log`.

Update the checkout and rerun the original command:

```bash
git pull --ff-only
./stack run recipes/flash17-libertai-exl3-2.9bpw.json --setup
```

The runtime ID change rebuilds the executable
and selects fresh asset/session namespaces; completed model downloads are reused. Host tests do not
establish live generation, memory fit or maximum-context GPU qualification.

## Origin-matched serving profiles (2026-10-26)

The [recipe settings audit](RECIPE-SETTINGS.md) records pinned source launchers, model cards, native
environment readers and the mappings used. The two 2.9 bpw recipes now use their source stacks' serving
choices instead of inheriting the entire 2.8 bpw profile. The baseline's SSD budget/minimum entry size
now match its source environment; earlier claims below that its recipe was unchanged are historical.

- `./scripts/test.sh`: 81 tests passed, plus shell syntax checks and the isolated baseline dry-run.
- Separate `./stack run NAME --setup --dry-run -n 192.0.2.11,192.0.2.12` checks passed for all three recipes.
  Both ranks receive the same slot count and unchanged FP8 index precision; the head receives the intended
  sampling flags. Distinct model/profile combinations produce distinct asset namespaces.
- New command-generation tests cover eight-slot propagation to both ranks, sampling defaults including
  top-k zero, configuration-label changes, unchanged kernel identity for sampling-only changes, and invalid
  sampling values. Existing checkpoint/cache, transfer, cancellation and lifecycle tests also passed.

No runtime source or patch was changed. These checks did not access either Spark, prepare GPU kernels,
download weights, deploy, or run generation. GPU memory fit, AOT coverage, eight-slot operation and
performance of the translated profiles remain unqualified. Prepare fresh assets with the selected recipe
before starting it; retain the previous assets/runtime for recovery.

## Multicore head preparation (2026-10-26)

Runtime `40776fc247332549` adds patch 0006. Shared AOT planning/checking uses eight CPU workers on
the head; the worker consumes head-generated assets with full generated-file SHA256 validation.
Rank-specific Engram packing/checks remain local. Model and KV precision are unchanged.

- 78 orchestration tests passed, including no worker enumeration/compilation, rank-local Engram checks,
  copied cubin/table corruption, missing cubins, wrong runtime identity, missing head receipts, and
  refusing to recertify corrupted cached head assets.
- The optimized host executable build passed all 30 steps. The combined host suite passed all 102
  steps (179 tests executed successfully, five skipped; other test results reused the build cache).
  The first sandboxed run encountered blocked local sockets; the successful rerun allowed loopback.
- Thread merge tests compare serial with 2/4/8/32 workers, including variant order, per-variant launch
  counts, program counts, missing coverage, and joining/propagating a failed worker without partial success.
- Full 1,048,576-token, four-slot baseline: the eight-worker output exactly matches the original head's
  serial JSON: 343,978 programs, 144 variants, identical variant order and every launch count. Coverage
  passed against the head's existing 144-variant AOT manifest. Recipe and signature file SHA256 matched
  across the reference and test inputs. The local x86_64 pass took 282.5 seconds; no comparable serial
  timing was completed locally, so this is not a measured Spark speedup. A redundant local serial run
  was stopped after retrieving the head's completed original serial reference.
- Fresh source assembly applied the full patch set without fuzz, matched the tested source, and passed
  repeat-fetch verification. Shell checks and isolated synthetic-node launch generation passed.

This is CPU preparation validation, not GPU inference or an end-to-end startup benchmark. This validation
did not deploy to the Sparks or interrupt their ongoing setup. Raw local evidence is in
`.stack/parallel-python-final.log`, `.stack/parallel-build.log`, `.stack/parallel-host-allowed.log`,
`.stack/parallel-fetch.log`, `.stack/parallel-covered.log`, and `.stack/parallel-equivalence.json`.

## Experimental 2.9 bpw recipes

Runtime `2c3def4c7fdda912`, tested locally on x86_64 with Zig 0.17.0:

| Check | Result |
|---|---|
| Utilities | 73 tests passed; shell checks and isolated dry-runs passed for all three recipes |
| Runtime host suite | Fresh cache: 103/103 build steps; 669 passed, 13 skipped, zero failures |
| Host executable build | 30/30 steps; no CUDA fatbins or GPU execution |
| Real checkpoint headers | All 43 layers planned on both TP=2 ranks for each pack; 47/47 loader tests passed per pack |
| Source assembly | Both new patches applied without fuzz; assembled and tested trees match; repeat fetch verified |
| Baseline recipe | Byte-for-byte unchanged from `main` at `a53e15a` |

Tests cover FP16 widening without BF16 rounding, unchanged FP32/BF16 outputs, invalid dtype/shape rejection,
pack-derived AOT widths, pinned Hub hashes, corruption, explicit tokens, and offline worker verification.
Setup/download now skip tensor hashing and use size-based cache reuse; tests prohibit default payload hash reads,
check explicit verification on both nodes, and retain missing/truncated-file checks. `verify-weights` checks
existing caches without downloads; `download --verify-weights` also enables rsync checksum repair.
Header checks used all 39 EAQ v1 and 43 LibertAI shard headers and their pinned indexes, without weight payloads.
The local suite requires loopback sockets for synthetic transport tests. No Sparks were accessed.

**Before merging:** build and prepare the candidate runtime on ARM64/GB10, qualify the original 2.8 bpw recipe
first, then each new recipe. Verify both ranks and generation, DSpark, memory under representative concurrent
long prompts, and actual AOT coverage. The new packs add about 6.5 GiB of weights across the pair. Host tests
do not establish GPU kernel correctness, memory fit, maximum context, modalities, quality, or performance.
Runtime changes update `spark-ds41-tensorfold:latest` and create a new asset namespace.
Launch receipts pin immutable image IDs; builds retain no separate rollback image.

## Earlier ARM64 baseline

The results below cover CPU and utility checks on two ARM64/GB10 nodes.
**A fresh deployment from this standalone repository remains to be qualified.**
Runtime ID: `f9361e2c7b98a65d`. Pins are in [SOURCES.md](SOURCES.md).

| Check | Result |
|---|---|
| Orchestration tests | 43 passed on development and ARM64/GB10 systems; shell checks and isolated dry-run passed |
| Standalone Git clone | 43 tests passed without local configuration or generated state; fresh public-source fetch and repeat-fetch verification passed |
| Source assembly | Pinned exports and all runtime/Python patches applied |
| Live setup | Two RoCE lanes discovered; saved configuration, doctor, and launch dry-run passed |
| Authentication | Each node's HF_TOKEN verified, including head-to-worker SSH |
| Cache reuse | Engram shards hash-verified on both nodes; Hub dry-runs required zero downloads for 51 model and four Engram files |
| Runtime host tests | 103/103 build steps succeeded; 641 passed, 12 skipped, zero failures |
| HTTP corpus | 312/319 matched; seven known differences, zero unexpected differences |
| Image fixtures | WebP 49/49; vision and vision-bias 54/54 each; fetch decisions 649/649 |
| Sampling | 207/207 rows matched Python |
| Asset smoke | Both synthetic Engram ranks matched source bytes; corruption rejected; 88 Triton signatures imported |

Runtime tests used an ARM64 Zig 0.17.0 builder, a fresh test cache, four CPUs, and 16 GiB RAM.
Networking and GPUs were disabled.
The seven known HTTP differences concern structured-output errors and metrics. Skipped tests remain unqualified.

Asset tests used the pinned NGC image with Triton 3.7.1 and ptxas 13.3.73, four CPUs, 8 GiB RAM, and no network
or GPU access. These checks did not compile the full AOT set or generate full-model RoPE/host tables.

## Reproduce

The head-only download/fabric-sync change passed 54 local orchestration tests, shell checks, and the isolated
two-node dry-run on 2026-10-26. Added checks include real rsync of small synthetic HF caches (snapshot links,
shared blobs, same-size corruption repair, repeat transfers, and unrelated-file preservation), offline worker
verification and receipts, insufficient disk space, and mocked fabric/SSH/Docker failures. They also check that
only the head downloads weights and pulls the preparation image, and that copied image IDs are verified.
No production weights/images were downloaded or transferred, and no Spark service was started for this change.
Live fabric throughput and a fresh deployment remain unqualified. Earlier 43-test results above predate this change.

Cancellation handling subsequently passed **62 local tests**, shell checks, and the isolated two-node dry-run
on 2026-10-26. Eight additional tests use real local subprocesses and pipes, including a simulated Hub client
through the actual download wrapper: repeated Ctrl+C, a child ignoring SIGTERM, parent death/connection EOF,
the embedded worker bootstrap without a checkout, parallel-call cancellation, and image-pipeline interruption
and failure. They verify bounded shutdown, child exit, reusable node locks, retained partial files, and absence
of an interrupted download receipt. These checks use no SSH, Docker, model downloads, or GPU workloads;
live Hugging Face/Xet cancellation and network-failure behavior on the Sparks remain unqualified.

```bash
./scripts/test.sh
./stack run flash17-exl3-2.8bpw --setup --dry-run -n 192.0.2.11,192.0.2.12
```

On an ARM64 Spark with fetched sources and the asset image available:

```bash
./stack fetch
./stack test-runtime
./stack test-assets
```

Host tests build and remove a temporary builder image, including after failure or cancellation.
Use `--builder-image IMAGE` to supply an existing image that the tests will leave intact.
Test evidence stays in ignored `.stack/`.

## Fresh deployment test

Use a new checkout and an empty worker directory after the existing GPU workload has been stopped separately.
Do not copy `cluster.json` or `.stack/` from an earlier installation. Keep the normal Hugging Face cache;
the test validates fresh setup and generated state without downloading cached weights again.

```bash
git clone https://github.com/hacibeyoglu/deepseek-v4.1-flash-tensorfold-spark.git model-stack-test
cd model-stack-test
./scripts/test.sh
./stack setup --worker worker-host
./stack run flash17-exl3-2.8bpw --setup
./stack verify
```

The worker directory defaults to the new checkout name. Verify both ranks, image IDs, model identity, and
generation. GPU build/preparation correctness, SSD-session replay, maximum context, sustained concurrency,
modalities, and performance need separate qualification; CPU tests do not establish those results.
