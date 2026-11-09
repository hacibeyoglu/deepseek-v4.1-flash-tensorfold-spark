# Optional in-engine compute

Reviewed 2026-11-09 against upstream recipe `24a84eb2cc81e297b827ba71126dafd5d755c561` and native
engine `7b79ef92d846138d51638b677c473392a56e240c`.

The standard recipes retain their existing serving settings and explicitly disable the sandbox lane,
default tool injection, runner-first tool execution and in-thinking runner. The engine update still
includes the NVMe session-restore fix for replay sessions rejected with `BadBlob`.

`flash17-jay-exl3-2.8bpw-compute` is an opt-in copy of the standard Jayleaton recipe with upstream's
nineteen compute settings. It keeps the checkpoint, precision, context, slot count, KV pool and other
serving settings. It leaves `TF_CUDA_SCHED` unset to match upstream's experimental measured profile;
the standard recipe retains `blocking`.

## Behavior

The compute recipe runs cells found in the model's thinking through the circuit and graph bank,
and offers the built-in `python(code)` tool to eligible thinking chats. Runner-first tool execution
falls back to the sandbox when the runner abstains. The default adds the runner's trust guide to
every thinking chat; tool admission additionally requires at least 1024 output tokens and no stop
strings, logprobs or structured output. Client function tools still return to the client; a client
function named `python` retains that name and suppresses the built-in tool for that request.
`runtime.compute_executed` counts execution, not merely a declaration.

The Python sandbox requires Landlock ABI 6 and seccomp, runs as nobody and disallows network and child
processes. Its read/execute boundary includes `/usr`, the interpreter install root and specific runtime
files; see upstream's full [operations description](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/24a84eb2cc81e297b827ba71126dafd5d755c561/docs/OPERATIONS.md).
The engine checks sandbox availability at boot. The runner frontend uses clang/lld 18 to compile its
circuit programs, which run through the engine's runner.

This remains experimental. Upstream's AB3 study failed its primary comparison and safety/provenance
qualification; it reported a trace newline recorded but not delivered. Later commits address that
delivery issue, but upstream explicitly does not claim a replacement full qualification. Its reported
token savings are workload-specific, and have not been measured on this environment. Sources:
[compute-lane study and limits](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/24a84eb2cc81e297b827ba71126dafd5d755c561/docs/compute-lane.md),
[engine and restore evidence](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/24a84eb2cc81e297b827ba71126dafd5d755c561/docs/ENGINE.md).

## Build and select a recipe

All recipes share one runtime image. It contains the compiler, circuit and graph bank even while
compute is disabled. The head downloads the release archive pinned in `config/runner-assets.release`;
both its SHA-256 and all 196 files in `config/runner-assets.sha256` are checked before image assembly.
The Docker build and `./stack build` image checks require `tf-runner-frontend-check --require` to pass.
Only the resulting image is transferred to the worker, with the existing image-ID verification.

Source and runner-input changes invalidate the runtime and asset namespaces. Existing downloaded
weights and prepared weight files remain reusable; a previous preparation receipt does not qualify
the new engine. On the configured head, during an intentional maintenance window:

```bash
./stack stop
./stack run flash17-jay-exl3-2.8bpw --setup
```

That updates the standard profile with compute off. To opt into the experimental profile instead:

```bash
./stack stop
./stack run flash17-jay-exl3-2.8bpw-compute --setup
```

Use the same selected recipe for later commands, for example
`./stack verify --recipe flash17-jay-exl3-2.8bpw-compute` and
`./stack start --recipe flash17-jay-exl3-2.8bpw-compute`.
Switch back with the standard recipe and matching preparation. No shell-exported compute settings
are inherited: the selected recipe determines both ranks' environment.

For command inspection without host discovery or changes:

```bash
./stack run flash17-jay-exl3-2.8bpw-compute --setup --dry-run -n 192.0.2.11,192.0.2.12
```

Local host tests and command generation do not establish ARM64 image build success, TP2 generation,
compute execution, long-context capacity or performance. See [validation](VALIDATION.md) for the actual
checks performed for this update.
