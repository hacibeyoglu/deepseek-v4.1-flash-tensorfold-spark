**Acknowledgements:** This project builds on the wonderful work of [ashhart/TensorFold](https://github.com/ashhart/TensorFold),
[jayleaton](https://github.com/jayleaton/deepseek-v41-tensorfold-spark),
[0rand](https://github.com/0rand/deepseek-v41-tensorfold-spark/tree/main-0rand), and
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).
# DeepSeek V4.1 Flash TensorFold for Spark

Run **DeepSeek-V4.1-Flash-1017 EXL3 2.8 bpw** with TensorFold's native Zig/CUDA runtime on
**two DGX Sparks, one GB10 GPU per node (TP=2)**. The head discovers the nodes, downloads weights,
builds and distributes images, prepares assets, and manages both ranks.

## Quick start

Both nodes need ARM64/GB10, Docker with NVIDIA Container Toolkit, working RoCE, Python 3.12+,
Git, `patch`, and rsync. The head also needs `venv`, `HF_TOKEN`, and passwordless SSH to the worker's
fabric address. Allow approximately **650 GiB of disk space per node** for a fresh installation.
The worker needs no Hugging Face token or manual checkout.

Run on the head:

```bash
git clone https://github.com/hacibeyoglu/deepseek-v4.1-flash-tensorfold-spark.git
cd deepseek-v4.1-flash-tensorfold-spark
./stack setup --worker worker-host
./stack doctor
./stack run flash17-jay-exl3-2.8bpw --setup
```

Replace `worker-host` with an SSH alias or `user@worker-host`. The worker directory defaults to
`~/<head checkout name>`; override it with `setup --worker-dir /path/to/worker-checkout`.
Bulk transfers require verified fabric SSH. Stop existing inference workloads before installation.

API: **`http://<head-host>:8000/v1`**, model **`deepseek-v4.1-flash`**.
Use `--host 127.0.0.1` for local-only access.

The first startup converts weights and can take several minutes. Subsequent launches reuse prepared
weights: use `./stack start`. For individual build stages, updates, cache behavior and troubleshooting,
see the [build and preparation guide](docs/BUILD.md).

## Everyday commands

Run these on the head to manage both nodes:

| Command | Purpose |
|---|---|
| `./stack start` | Start a prepared stack and verify generation |
| `./stack stop` | Drain and stop both ranks |
| `./stack restart` | Recreate both ranks with current settings |
| `./stack status` | Inspect both ranks |
| `./stack logs --rank both --tail 100` | Read recent logs |
| `./stack doctor` | Check nodes, network and containers |
| `./stack verify` | Verify images, settings, model, health and generation |
| `./stack run --list` | List available recipes |

Both nodes must be reachable. Lifecycle checks reject conflicting workloads and partial pairs.
Both ranks use `unless-stopped`; an intentional stop disables automatic restart. Failed startup
prints both ranks' recent logs, stops newly launched ranks and retains their data.

## Configuration

Portable model settings live in [recipes](recipes); local paths and topology live in ignored
`cluster.json`. Weights use the normal Hugging Face cache on each node; generated assets and state
live in ignored `.stack/`. Downloads check file sizes by default; use `./stack verify-weights`
for explicit checksum verification.

`./stack fetch` checks out the pinned native engine and applies the reviewed engine patch series in
[patches/engine](patches/engine) with `git am`, then exports the asset-generator references. It then applies
the three Python patches for the TensorFold reference. The seven older runtime patches in [patches](patches)
are retained as `legacy_patches` and are not applied. Upstream removed the Python patches from its own tree,
so they are read from the commit recorded under `python_patches` in [sources.lock.json](sources.lock.json)
and each file is checked against its SHA-256 before it is applied. Every engine patch is hash-checked, the
series must reproduce upstream's engine tree, and the assembled content must match the lock. See
[source provenance](docs/SOURCES.md).

The default profile uses four slots, 1,048,576 context per request, a shared 2.1M-token KV pool,
FP8 index KV, DSpark, native images and SSD session reuse. These are configured capacities;
see [validation limits](docs/VALIDATION.md) for tested workloads.

The native engine is upstream TensorFold `f212eeabd` plus the reviewed 91-patch series that reproduces
the engine upstream serves: the shared compute lane and in-thinking runner, NVMe session restore,
session checkpoints and undeclared tool-call passthrough, the Qwen lane fixes (`0073`-`0087`), DSpark
staging alignment (`0088`), the optional adaptive context window (`0089`), replay-prefill short-prompt
repetition (`0090`) and the reasoning-only trace close (`0091`). Computation stays **off** in the
existing recipes. The optional `flash17-jay-exl3-2.8bpw-compute` recipe adopts upstream's
experimental in-thinking compute profile. It can execute built-in Python calls and runner cells inside
the engine; see [compute behavior and limits](docs/COMPUTE.md) before selecting it.

Additional recipes cover Mia EAQ v1 and LibertAI EXL3 2.9 bpw. Mia remains experimental;
LibertAI has limited TP=2 text validation. Supporting other model families requires runtime work.
See [recipe settings](docs/RECIPE-SETTINGS.md) for exact defaults and differences.

Use `--recipe NAME` to select a recipe. Changes to the model, runtime, context, slots or environment
require matching preparation before startup; follow [configuration changes](docs/BUILD.md#recipe-and-capacity-changes).
The recipe controls runtime variables; shell-exported TensorFold settings are not used.

## Documentation

- [Build and preparation](docs/BUILD.md): staged builds, updates, downloads, storage and capacity changes.
- [Source provenance](docs/SOURCES.md): upstream pins, local patches and branch reviews.
- [Recipe settings](docs/RECIPE-SETTINGS.md): serving profiles and source mappings.
- [Validation](docs/VALIDATION.md): checks, measured results and remaining limits.

## License

Apache-2.0. Fetched sources retain upstream notices; model weights retain their own licenses.
