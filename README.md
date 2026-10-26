# DeepSeek V4.1 Flash TensorFold for Spark

**Acknowledgements:** Thanks to these projects and their contributors:

- [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark): runtime integration, model tooling, and Docker build foundation.
- [0rand/deepseek-v41-tensorfold-spark (pr65-flash17-serving)](https://github.com/0rand/deepseek-v41-tensorfold-spark/tree/pr65-flash17-serving): session persistence, memory handling, and telemetry improvements.
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker): workflow reference for head-managed workers and recipe utilities.

A two-Spark stack for **DeepSeek-V4.1-Flash-1017 EXL3 2.8 bpw**, using TensorFold's native Zig/CUDA runtime.
**TP=2 (tensor parallelism): two nodes with one GB10 GPU each, one head and one worker.**
The head discovers both nodes, downloads weights once, syncs weights and Docker images over the fabric,
prepares assets, and manages both ranks.
Supporting other model families requires runtime and preparation changes.
See [source provenance](docs/SOURCES.md) and [validation limits](docs/VALIDATION.md).

## Quick start

Clone this repository on the head. Both nodes need ARM64/GB10, Docker with NVIDIA Container Toolkit, working RoCE,
Python 3.12+, Git, `patch`, and rsync. The head also needs `venv` and `HF_TOKEN`, plus passwordless SSH to the
worker's fabric address. The worker needs no Hugging Face token or download client.

```bash
git clone https://github.com/hacibeyoglu/deepseek-v4.1-flash-tensorfold-spark.git
cd deepseek-v4.1-flash-tensorfold-spark
./stack setup --worker worker-host     # Once: discover nodes and save cluster.json
./stack doctor
./stack run flash17-exl3-2.8bpw --setup   # Fetch, download, build, prepare, and start

./stack status
./stack logs --rank both --tail 100
./stack stop
./stack start                       # Subsequent starts
```

Replace `worker-host` with your worker's SSH alias or `user@worker-host`. Choose any head checkout location;
the worker directory defaults to `~/<head checkout name>`, or set `--worker-dir /path/to/worker-checkout`.
Bulk transfers require verified fabric SSH. No manual worker checkout is needed.

API: `http://<head-host>:8000/v1`, model `deepseek-v4.1-flash`. Use `--host 127.0.0.1` for local-only access.

## Commands

Run `./stack` commands on the head to manage both nodes. All shell utilities live in `scripts/`;
for example, `./scripts/start.sh` is equivalent to `./stack start`.

| Command | Purpose |
|---|---|
| `./stack setup --worker HOST` / `./stack doctor` | Discover configuration / recheck nodes, network, and containers |
| `./stack fetch` | Export pinned sources and apply patches |
| `./stack download` | Download and hash-check weights/Engram shards on the head; sync and verify the worker offline |
| `./stack build` | Build images and verify the same runtime image on both nodes; requires fetched sources |
| `./stack prepare` | Pull the preparation image on the head and copy it; pack Engram, generate assets, and check AOT coverage |
| `./stack start` / `./stack stop` | Start worker then head and verify generation / drain and stop both ranks |
| `./stack restart` / `./stack rollback` | Recreate with current settings / restore the configuration saved before restart |
| `./stack status` / `./stack logs --rank both` | Inspect state and logs |
| `./stack verify` | Check image, effective settings, model, health, and generation |
| `./scripts/test.sh` | Run local tests and an isolated dry-run; runtime/asset checks are in [VALIDATION.md](docs/VALIDATION.md) |

`./stack run --list` lists recipes. `--download-only`, `--build-only`, and `--prepare-only` select individual
stages; `--build-only` also fetches sources. Plain `./stack run flash17-exl3-2.8bpw` starts an already-prepared stack.

## Storage and authentication

Weights stay in the normal **`~/.cache/huggingface/hub`** on each node. Discovery honors `HF_HOME`, `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, and `XDG_CACHE_HOME`, saving each node's resolved paths in ignored `cluster.json`.
Containers mount the entire Hub cache read-only so snapshot symlinks work. Cached blobs are reused.

`./stack download` downloads the pinned model and original Engram shards on the head only. It then uses rsync
over SSH to the worker's first matched fabric address, preserving snapshot links and their required blobs
(including hub-level shared blobs). Only the selected snapshots and referenced blobs are copied; other caches
and revisions are retained. Transfers use checksums, partial-file reuse, and atomic replacement. The worker
verifies SHA256 hashes offline before writing its download receipt. Repeating the command reuses cached data;
checksum scans still read the selected files on disk.

`./stack build` builds the runtime image on the head and streams `docker save` into worker `docker load`.
`./stack prepare` pulls the pinned preparation image on the head and distributes it the same way. Matching
image IDs skip transfer, and both sides' IDs are checked afterward. Preparation runs by the resolved image ID
because Docker archives do not preserve registry digest references.

Both bulk paths verify node identity and routing over the saved RoCE interfaces, and fail if fabric SSH is
unavailable instead of falling back to management Ethernet. These are **SSH/TCP transfers over the RoCE
network**, not RDMA file transfers; they use one fabric lane. Run these stages with this stack stopped.

All other generated files belong under the checkout's ignored **`.stack/`**: source exports, build files,
Python environment, prepared assets, packed Engram files, sessions, calibration, and request logs.
Docker uses its standard image/build/log storage. A fresh installation can need approximately **650 GiB per node**.
Disk space is checked before large operations; caches are never pruned automatically.

Downloads explicitly use the head's **`HF_TOKEN`**. If absent from its environment, the downloader reads the
head's existing interactive Bash export. It verifies authentication before downloading and stops on failure.
Tokens are never logged, stored in stack configuration, or copied between nodes. Check head authentication after
installing the download client with:

```bash
.stack/venv/bin/python lib/hub.py --check-auth
```

## Configuration

`recipes/flash17-exl3-2.8bpw.json` holds portable model settings; `cluster.json` holds local paths and topology. Source and
model revisions are pinned. To use another recipe, pass `--recipe path/to/recipe.json`.

```bash
# Isolated plan: no discovery, network calls, downloads, or writes
# Replace <HEAD_IP> and <WORKER_IP> with IPv4 addresses, head first.
./stack run flash17-exl3-2.8bpw --setup --dry-run -n "<HEAD_IP>,<WORKER_IP>"
./stack start --dry-run             # Plan using saved cluster configuration

# Apply matching overrides during preparation and startup
./stack prepare --context 131072 --slots 4 --pool-tokens 1000000
./stack start --context 131072 --slots 4 --pool-tokens 1000000 --port 8000
```

Model, runtime, context, slot, or environment changes require preparation in a new asset/session namespace.
Stop the stack before preparation. Port/host changes only require recreation. The recipe controls runtime
variables; shell-exported TensorFold settings are not used.

Defaults: four slots, 1,048,576 context per request, a **shared 2.1M-token KV pool**, FP8 index KV, expert top-p
`0.85`, DSpark, native images, structured output, and a 32-GiB SSD session tier with immediate write-through.
Memory checks retain the 5-GiB admission floor, 4-GiB hard floor, 3.5-GiB prefill reserve, measured boot guard,
and 4K-to-2K prefill fallback. The launcher does not silently change precision or capacity.

## Lifecycle checks

Both nodes must be reachable. Start checks ownership, assets, images, ports, and GPU availability; it refuses
partial pairs or conflicting workloads. Preparation refuses to change assets mounted by running ranks.
Worker synchronization preserves `.stack/` and requires an empty directory or one already owned by this stack.

Both ranks use `unless-stopped`. Stop disables restart on both before draining. Failed startup stops newly
launched ranks and retains their data. Restart saves a rollback receipt; rollback is explicit. Operations are locked.
Successful readiness checks both ranks, API model identity, health, and actual generation. Maximum context,
concurrency, modalities, and performance need separate qualification.

## License

Apache-2.0. Fetched sources retain upstream notices; model weights retain their own licenses.
