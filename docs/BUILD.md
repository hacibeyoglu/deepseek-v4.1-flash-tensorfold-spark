# Build and preparation guide

Run commands from the checkout on the head. For initial installation prerequisites and cloning,
see the [README](../README.md). Exact revisions and patch explanations are in
[source provenance](SOURCES.md); measured results and qualification limits are in [validation](VALIDATION.md).

## Build in stages

After `./stack setup --worker worker-host` and `./stack doctor`, use the following stages.
Keep this stack stopped during downloads, image builds/transfers and asset preparation.
Use the same `--recipe NAME` on every stage when selecting a non-default recipe.

```bash
./stack stop             # If this stack is running
./stack fetch            # Export pinned sources and apply patches
./stack download         # Download on the head; sync selected files to the worker
./stack build            # Build on the head; copy and verify the worker image
./stack prepare          # Prepare shared assets and rank-specific Engram files
./stack start            # Start both ranks; verify readiness and generation
```

For a fresh installation, `./stack run flash17-jay-exl3-2.8bpw --setup` performs the complete sequence.
`./stack run --list` lists recipes. `--download-only`, `--build-only`, and `--prepare-only` select
individual stages; `--build-only` also fetches sources. Plain `./stack run flash17-jay-exl3-2.8bpw`
starts an already-prepared stack. All shell wrappers live in `scripts/`.

## Updating runtime sources

Review upstream changes and preserve local edits before changing [sources.lock.json](../sources.lock.json)
or [patches](../patches). Follow the source review process in [SOURCES.md](SOURCES.md).
After editing orchestration, run `./scripts/test.sh`; runtime patches also require the combined host tests.
With the stack stopped:

```bash
./stack fetch
./stack test-runtime
./stack build
./stack prepare
./stack start
```

Run `./stack test-assets` when validating asset tooling; it requires the preparation image to be available.
Host tests and isolated dry-runs do not establish GPU correctness, memory fit, maximum context,
modalities or concurrent performance. See [validation procedures](VALIDATION.md#reproduce).

## Downloads, storage and authentication

Weights stay in the normal **`~/.cache/huggingface/hub`** on each node. Discovery honors `HF_HOME`, `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, and `XDG_CACHE_HOME`, saving each node's resolved paths in ignored `cluster.json`.
Containers mount the entire Hub cache read-only so snapshot symlinks work. Cached blobs are reused.

`./stack download` downloads the pinned model and original Engram shards on the head only. It then uses rsync
over SSH to the worker's first matched fabric address, preserving snapshot links and their required blobs
(including hub-level shared blobs). Only the selected snapshots and referenced blobs are copied; other caches
and revisions are retained. Transfers use size-based cache reuse, partial-file reuse, and atomic replacement.
The worker checks file presence, sizes, and snapshot links before writing its download receipt.
**Setup and downloads do not scan tensor checksums by default.** Same-size corruption requires explicit verification:

```bash
./stack verify-weights --recipe flash17-jay-exl3-2.8bpw    # Check existing caches on both nodes
./stack download --verify-weights                   # Download/sync with checksum scans
# Also supported: ./stack run RECIPE --setup --verify-weights
```

Select the same recipe used for downloading. `verify-weights` reports corruption without changing files;
`download --verify-weights` also uses rsync checksums to repair worker files from a valid head cache.
All rsync stages show filenames, aggregate transfer progress (bytes, percentage, speed, and time estimates),
and final statistics, with line-buffered output. Explicit checksum verification reads the full selected weights
and can take time even when the cache is already populated. API readiness and generation checks remain automatic.

## Runtime images and reuse

`./stack build` builds the runtime image on the head and streams `docker save` into worker `docker load`.
All recipes use **`spark-ds41-tensorfold:latest`**, the only retained stack runtime tag.
Builds skip images whose runtime fingerprint already matches the source lock, patches, and Dockerfile.
Updating those inputs replaces the same tags. Recipe changes reuse the runtime and select their own assets.
The fingerprint remains in image labels and asset paths; preparation/start reject a stale runtime.
Launches pin exact image IDs. `./stack test-runtime` builds a temporary compiler/test image and removes it
after success, failure, or cancellation. `--builder-image IMAGE` uses an existing caller-owned image and
leaves it intact. Docker build cache remains available for faster rebuilds; no global pruning is performed.
`./stack prepare` pulls the pinned preparation image on the head and distributes it the same way. Matching
image IDs skip transfer, and both sides' IDs are checked afterward. Preparation runs by the resolved image ID
because Docker archives do not preserve registry digest references.

## Prepared weights and AOT assets

The first API startup also converts checkpoint tensors into native per-rank weight images (about 95 GiB
per node). This can take several minutes after Docker starts; it is separate from Engram packing and AOT
preparation. Let it finish once, then use `./stack start` for subsequent launches instead of repeating
`run --setup`. Prepared images stay in `.stack/prepared/<model-revision>/` across runtime updates.
TensorFold's inner cache key still checks the prepared format, weight-builder sources, checkpoint shard
metadata, TP layout and DSpark; incompatible images are rebuilt. Existing runtime-specific caches are
adopted automatically by hard link during preflight, without copying or hashing the weight payloads.
Completed files and old cache paths are preserved; interrupted `.tmp` files are ignored. A runtime
update still needs its matching AOT assets, but an unrelated runtime change need not reconvert weights.

Shared assets are prepared on the head only: AOT requirement enumeration and coverage checking use
**eight CPU workers** for independent decode scans and prefill rows. The head compiles the Triton kernels,
generates RoPE and Engram host lookup tables, then copies those assets over fabric SSH. The worker checks
the runtime/asset identity and SHA256 of every shared file (including cubins) against the head's receipt;
it does not repeat AOT enumeration, compilation, or shared table generation. Each rank still packs and
checks its own Engram files. These small/generated-asset hashes do not hash the model weights.

For serial diagnostics, set `TF_DSV41_AOT_WORKERS` to `"1"` in a custom recipe's `env`; valid values are
1–32, with 8 as the runtime default. This controls CPU planning only, not inference precision or GPU
parallelism. Adaptive regime discovery and some replay checks remain serial; speedup depends on CPU
and workload. After updating runtime patches, use `./stack fetch`, `./stack build`, and `./stack prepare`
with the stack stopped; the changed runtime gets a new asset namespace. Existing runs are not modified.

## Transfer requirements and disk layout

Both bulk paths verify node identity and routing over the saved RoCE interfaces, and fail if fabric SSH is
unavailable instead of falling back to management Ethernet. These are **SSH/TCP transfers over the RoCE
network**, not RDMA file transfers; they use one fabric lane. Run these stages with this stack stopped.

All other generated files belong under the checkout's ignored **`.stack/`**: source exports, build files,
Python environment, prepared assets, packed Engram files, sessions, calibration, and request logs.
Docker uses its standard image/build/log storage. A fresh installation can need approximately **650 GiB per node**.
Disk space is checked before large operations; caches are never pruned automatically.

## Authentication

Downloads explicitly use the head's **`HF_TOKEN`**. If absent from its environment, the downloader reads the
head's existing interactive Bash export. It verifies authentication before downloading and stops on failure.
Tokens are never logged, stored in stack configuration, or copied between nodes. Check head authentication after
installing the download client with:

```bash
.stack/venv/bin/python lib/hub.py --check-auth
```

## Recipe and capacity changes

`recipes/flash17-jay-exl3-2.8bpw.json` holds portable model settings; `cluster.json` holds local paths and topology. Source and
model revisions are pinned. To use another recipe, pass `--recipe path/to/recipe.json`.

Two additional recipes are available; Mia remains experimental, while LibertAI has limited TP=2 text validation:

| Recipe | Checkpoint |
|---|---|
| `flash17-mia-exl3-2.9bpw` | Mia EAQ v1, EXL3 2.9 bpw |
| `flash17-libertai-exl3-2.9bpw` | LibertAI, EXL3 2.9 bpw |

Use either name with `./stack run NAME --setup`. Both use 2K prefill and full expert top-k routing. Mia keeps
four slots, a replicated 1,048,576-position pool and RAM-only session reuse. LibertAI uses four slots,
262,144 context per request and a shared split 524,288-position pool, with buffer reuse and a 32-GiB SSD tier
to fit the native runtime's memory budget. Startup and four fresh approximately 12K-token text requests passed;
maximum context, image/tool behavior and performance remain unqualified. See [recipe settings and origin
mappings](RECIPE-SETTINGS.md) for values and [validation](VALIDATION.md) for measured limits.
Both preserve the existing precision and memory guards. They use pinned Hub hashes because these packs omit `SHA256SUMS`, and derive AOT
kernel requirements from their own headers. EAQ v1.1 and display-KV are not included.

```bash
# Isolated plan: no discovery, network calls, downloads, or writes
# Replace <HEAD_IP> and <WORKER_IP> with IPv4 addresses, head first.
./stack run flash17-jay-exl3-2.8bpw --setup --dry-run -n "<HEAD_IP>,<WORKER_IP>"
./stack start --dry-run             # Plan using saved cluster configuration

# Apply matching overrides during preparation and startup
./stack prepare --context 131072 --slots 4 --pool-tokens 1000000
./stack start --context 131072 --slots 4 --pool-tokens 1000000 --port 8000
```

Model, runtime, context, slot, or environment changes require preparation in a new asset/session namespace.
Stop the stack before preparation. Port/host and sampling-default changes only require recreation. Optional
`serve.temperature`, `serve.top_p`, and `serve.top_k` become server defaults; requests can override them.
The recipe controls runtime
variables; shell-exported TensorFold settings are not used.

Default 2.8 bpw profile: four slots, 1,048,576 context per request, a **shared 2.1M-token KV pool**, FP8 index KV,
expert top-p `0.85`, DSpark, native images, structured output, and a 128-GiB SSD session tier with immediate
write-through for entries of at least 1024 tokens, matching the source environment's budget and minimum size.
Memory checks retain the 5-GiB admission floor, 4-GiB hard floor, 3.5-GiB prefill reserve, measured boot guard,
and 4K-to-2K prefill fallback. The launcher does not silently change precision or capacity.

## Cancelling downloads and transfers

Downloads run in the foreground.Press Ctrl+C once to cancel: the launcher waits for its downloader and
transfer processes to stop, escalating to a forced stop if needed, then exits with status 130. Further Ctrl+C
presses during cleanup are ignored so children can be reaped and locks released. Completed cache files and
partial transfers remain for the next run; interrupted work does not produce a new success receipt.
Worker agents also cancel their child processes when the parent SSH connection closes. A broken or unreachable
worker is still reported as unknown; connection loss alone does not prove that remote work has stopped.
