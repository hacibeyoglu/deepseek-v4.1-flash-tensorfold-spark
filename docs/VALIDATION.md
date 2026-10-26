# Validation (2026-10-26)

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

On an ARM64 Spark with the builder and asset images available:

```bash
./stack fetch
./stack test-runtime --builder-image <builder-image>
./stack test-assets
```

The default builder is produced by `./stack build`. Test evidence stays in ignored `.stack/`.

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
