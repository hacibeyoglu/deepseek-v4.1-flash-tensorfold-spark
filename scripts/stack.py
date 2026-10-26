#!/usr/bin/env python3
"""Head-side CLI. Use --help or any of the convenience shell scripts."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import (StackError, load_recipe, plan, read_json, runtime_id,
                        synthetic_cluster, validate_recipe)
from lib.cluster import Cluster, setup
from lib import processes, sources

def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="action", required=True)
    for name in ("setup", "fetch", "download", "build", "prepare", "start", "stop", "restart",
                 "status", "logs", "doctor", "verify", "rollback", "plan", "run", "test-runtime", "test-assets"):
        sub = commands.add_parser(name)
        sub.add_argument("--recipe", default="flash17-exl3-2.8bpw", help="Recipe name or JSON path")
        sub.add_argument("--config", type=Path, default=ROOT / "cluster.json")
        sub.add_argument("--dry-run", action="store_true", help="Print only; never probe, download, or mutate")
        sub.add_argument("-n", "--nodes", help="Explicit head,worker IPv4 addresses for isolated dry-runs only")
        for key in ("port", "context", "slots", "pool-tokens"):
            sub.add_argument("--" + key, type=int)
        sub.add_argument("--host", choices=("0.0.0.0", "127.0.0.1"))
        if name == "setup":
            sub.add_argument("--worker", required=True, help="SSH alias or user@host reachable from the head")
            sub.add_argument("--worker-dir", help="Default: worker's ~/<this checkout name>")
        if name == "fetch":
            sub.add_argument("--source-cache", type=Path, help="Use an existing Git repo containing the pinned upstream commit")
            sub.add_argument("--python-cache", type=Path, help="Use an existing Git repo containing the pinned Python reference")
        if name == "logs":
            sub.add_argument("--rank", choices=("0", "1", "both"), default="both")
            sub.add_argument("--tail", type=int, default=100)
        if name == "run":
            sub.add_argument("recipe_name", nargs="?")
            group = sub.add_mutually_exclusive_group()
            for flag in ("setup", "download-only", "build-only", "prepare-only"):
                group.add_argument("--" + flag, action="store_true")
            sub.add_argument("--list", action="store_true")
        if name == "test-runtime":
            sub.add_argument("--builder-image", help="Default: this runtime's builder image")
    return ap


@contextlib.contextmanager
def locked():
    (ROOT / ".stack").mkdir(exist_ok=True)
    with (ROOT / ".stack/control.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise StackError("Another head-side stack operation is in progress") from exc
        yield


def runtime_tests(image):
    sources.verify(ROOT)
    cache = ROOT / ".stack/test-cache"
    cache.mkdir(exist_ok=True)
    # std.testing.tmpDir and the HTTP corpus create fixtures beneath the source cwd.
    # Use a disposable writable copy, keeping the reviewed export unchanged.
    test_source = cache / "source"
    shutil.copytree(ROOT / ".stack/source/engine/zig", test_source, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".zig-cache", "zig-out", "__pycache__"))
    import os
    sources.run(["docker", "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "16g",
                 "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp",
                 "--env", "NVIDIA_VISIBLE_DEVICES=void",
                 "--env", "ZIG_GLOBAL_CACHE_DIR=/cache/global",
                 "--mount", f"type=bind,src={test_source},dst=/src",
                 "--mount", f"type=bind,src={cache},dst=/cache", "--workdir", "/src",
                 "--entrypoint", "/opt/zig/zig", image, "build", "test", "test-golden", "-j4",
                 "--cache-dir", "/cache/local", "--prefix", "/tmp/out"])


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "run" and args.list:
        print("\n".join(p.stem for p in sorted((ROOT / "recipes").glob("*.json"))))
        return 0
    recipe = load_recipe(ROOT, getattr(args, "recipe_name", None) or args.recipe)
    for key in ("port", "context", "slots", "host"):
        value = getattr(args, key)
        if value is not None: recipe["serve"][key] = value
    if args.pool_tokens is not None:
        if args.pool_tokens < recipe["serve"]["context"]:
            raise StackError("--pool-tokens must cover at least one full context")
        recipe["env"]["TF_DSV41_POOL_TOKENS"] = str(args.pool_tokens)
    validate_recipe(recipe)
    if args.nodes and not (args.dry_run or args.action == "plan"):
        raise StackError("--nodes is for isolated plans only; use ./stack setup --worker for a real cluster")
    if args.dry_run or args.action == "plan":
        cluster = synthetic_cluster(ROOT, args.nodes) if args.nodes else read_json(args.config)
        generated = plan(ROOT, recipe, cluster)
        generated["action"] = args.action
        generated["steps"] = (["fetch", "download", "build", "prepare", "start", "verify"]
                              if getattr(args, "setup", False) else [args.action])
        print(json.dumps(generated, indent=2))
        return 0
    if args.action == "setup":
        with locked(): setup(ROOT, args.worker, args.worker_dir, args.config)
        return 0
    if args.action == "fetch":
        with locked(): sources.fetch(ROOT, args.source_cache, args.python_cache)
        return 0
    if args.action == "test-runtime":
        with locked(): runtime_tests(args.builder_image or f"spark-tensorfold:{runtime_id(ROOT)}-builder")
        return 0
    if args.action == "test-assets":
        import os
        with locked():
            sources.verify(ROOT)
            output = ROOT / ".stack/asset-test"
            output.mkdir(exist_ok=True)
            image = read_json(ROOT / "sources.lock.json")["asset_image"]
            sources.run(["docker", "run", "--rm", "--pull=never", "--network", "none", "--cpus", "4", "--memory", "8g",
                         "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp", "--env", "NVIDIA_VISIBLE_DEVICES=void",
                         "--env", "PYTHONDONTWRITEBYTECODE=1", "--mount", f"type=bind,src={ROOT},dst=/stack,readonly",
                         "--mount", f"type=bind,src={output},dst=/out", "--entrypoint", "python", image,
                         "/stack/tests/asset_smoke.py"])
        return 0
    cluster = Cluster(ROOT, recipe, read_json(args.config))
    cluster.require_head()
    if args.action == "status":
        result, ok = cluster.status()
        print(json.dumps(result, indent=2))
        return 0 if ok else 1
    if args.action == "logs":
        if args.tail < 1: raise StackError("--tail must be positive")
        cluster.logs(args.rank, args.tail)
        return 0
    if args.action == "doctor":
        print(json.dumps(cluster.doctor(), indent=2))
        return 0
    with locked():
        if args.action == "run":
            if args.setup:
                sources.fetch(ROOT)
                cluster.download()
                cluster.build()
                cluster.prepare()
                result = cluster.start()
            elif args.download_only: result = cluster.download()
            elif args.build_only:
                sources.fetch(ROOT)
                result = cluster.build()
            elif args.prepare_only: result = cluster.prepare()
            else: result = cluster.start()
        elif args.action == "rollback":
            previous = read_json(ROOT / ".stack/rollback.json")
            restored = Cluster(ROOT, previous["recipe"], previous["cluster"])
            # Saved concrete commands and asset paths allow rollback even after source-lock updates.
            restored.plan = previous["plan"]
            restored.nodes = previous["plan"]["nodes"]
            restored.preflight()
            cluster.stop()
            result = restored.start()
        else:
            result = getattr(cluster, args.action)()
        if result is not None: print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    processes.install_signals()
    try: sys.exit(main())
    except (StackError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"stack: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted; completed downloads/assets remain available for the next run.", file=sys.stderr)
        sys.exit(130)
