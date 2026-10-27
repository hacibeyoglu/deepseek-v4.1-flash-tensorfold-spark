"""All control runs on the head; workers receive explicit rank-local plans."""
from __future__ import annotations
import base64
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from .config import (StackError, atomic_json, choose_network, digest, plan, read_json,
                     safe_path, ssh_target, validate_cluster)
from . import cache_sync, processes, sources

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=3"]
RSYNC = ["rsync", "-av", "--protect-args", "--human-readable", "--info=progress2", "--stats", "--outbuf=L"]


class Remote:
    def __init__(self, root, node):
        self.root, self.node = Path(root), node

    def call(self, action, _cancel=None, **kwargs):
        payload = {"action": action, "root": self.node.get("root"), "watch_parent": True, **kwargs}
        if self.node.get("target"):
            script = base64.b64encode((self.root / "lib/node.py").read_bytes()).decode()
            helpers = base64.b64encode((self.root / "lib/processes.py").read_bytes()).decode()
            code = ("import base64,sys,types; m=types.ModuleType('processes'); "
                    f"exec(compile(base64.b64decode('{helpers}'), 'stack-processes', 'exec'), m.__dict__); "
                    "sys.modules['processes']=m; "
                    f"exec(compile(base64.b64decode('{script}'), 'stack-worker', 'exec'))")
            command = [*SSH, ssh_target(self.node["target"]), shlex.join(["python3", "-c", code])]
        else:
            command = [sys.executable, str(self.root / "lib/node.py")]
        proc = processes.run(command, input=json.dumps(payload) + "\n", text=True, stdout=subprocess.PIPE,
                             keep_stdin=True, cancel=_cancel, grace=6)
        try: response = json.loads(proc.stdout)
        except ValueError as exc:
            raise StackError(f"{self.node.get('target') or 'head'}: worker state is unknown (SSH/agent failed)") from exc
        if proc.returncode or not response.get("ok"):
            raise StackError(f"{self.node.get('target') or 'head'}: {response.get('error', 'agent failed')}")
        return response["result"]


def setup(root, worker, worker_root=None, config=None):
    root = Path(root).resolve()
    config = Path(config or root / "cluster.json")
    if config.exists():
        raise StackError(f"{config} already exists; edit it or use ./stack doctor to validate it")
    worker = ssh_target(worker)
    head = Remote(root, {"root": str(root)}).call("probe")
    peer = Remote(root, {"target": worker}).call("probe")
    if peer["identity"] == head["identity"]: raise StackError("The worker resolves to the head itself")
    if worker_root is None: worker_root = str(Path(peer["home"]) / root.name)
    if worker_root.startswith("~/"): worker_root = peer["home"] + worker_root[1:]
    safe_path(worker_root)
    net = choose_network(head, peer)
    cluster = {"schema": 1, "id": os.urandom(8).hex(), "network": net, "nodes": []}
    for rank, info in enumerate((head, peer)):
        cluster["nodes"].append({k: info[k] for k in ("hostname", "identity", "uid", "gid", "hf_home", "hub_cache")})
        cluster["nodes"][-1].update(rank=rank, root=str(root) if rank == 0 else worker_root,
                                    target=None if rank == 0 else worker)
    validate_cluster(cluster)
    for rank, node in enumerate(cluster["nodes"]):
        routes = [(net["nodes"][1-rank]["addresses"][i],
                   next(x["interface"] for x in (head, peer)[rank]["rdma"] if x["address"] == address))
                  for i, address in enumerate(net["nodes"][rank]["addresses"])]
        Remote(root, node).call("route", routes=routes)
    # Use a verified fabric SSH destination for large transfers when SSH is available there.
    fabric_target = peer["username"] + "@" + net["nodes"][1]["addresses"][0]
    try:
        probe = Remote(root, {"target": fabric_target}).call("probe")
        if probe["identity"] == peer["identity"]: cluster["nodes"][1]["target"] = fabric_target
    except StackError:
        print("Fabric SSH is unavailable; keeping the supplied SSH target for transfers.", file=sys.stderr)
    atomic_json(config, cluster)
    print(f"Saved {config}; {len(net['nodes'][0]['hcas'].split(','))} matched RoCE lane(s).")
    return cluster


def api(port, path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class Cluster:
    def __init__(self, root, recipe, cluster):
        self.root = Path(root).resolve()
        self.recipe = recipe
        self.config = validate_cluster(cluster)
        self.plan = plan(root, recipe, cluster)
        self.nodes = self.plan["nodes"]
        self.remotes = [Remote(root, n) for n in self.nodes]
        self.lock = read_json(self.root / "sources.lock.json")

    def require_head(self):
        node = self.nodes[0]
        if Path(node["root"]).resolve() != self.root or node["hostname"] != socket.gethostname():
            raise StackError("Run this command from the configured checkout on the head node")

    def request(self, rank):
        return {"node": self.nodes[rank], "recipe": self.recipe, "image": self.plan["image"],
                "runtime": self.plan["runtime"], "asset_key": self.plan["asset_key"],
                "asset_image": getattr(self, "asset_image", self.lock["asset_image"])}

    def pair(self, action, requests=None):
        # Read both results, including failures. Never reinterpret an unreachable worker as absent.
        cancel = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            try:
                jobs = [pool.submit(r.call, action, _cancel=cancel, **((requests or [{}, {}])[i]))
                        for i, r in enumerate(self.remotes)]
                results, failures = [], []
                for i, future in enumerate(jobs):
                    try: results.append(future.result())
                    except Exception as exc:
                        failures.append(f"rank {i}: {exc}")
                        results.append(None)
            except BaseException:
                with processes.cleaning_up():
                    cancel.set()
                    pool.shutdown(wait=True, cancel_futures=True)
                raise
        if failures: raise StackError("; ".join(failures))
        return results

    def states(self):
        return self.pair("state", [{"name": n["name"], "stack": self.config["id"], "expected_env": n["env"],
                                    "expected_command": n["command"][n["command"].index(self.plan["image"]) + 1:]}
                                   for n in self.nodes])

    def sync_code(self):
        peer = self.nodes[1]
        self.remotes[1].call("claim", stack=self.config["id"])
        args = [*RSYNC, "--exclude=.stack", "--exclude=.git", "--exclude=cluster.json*",
                "--exclude=__pycache__", "--exclude=.venv", "-e", shlex.join(SSH),
                str(self.root) + "/", peer["target"] + ":" + peer["root"] + "/"]
        sources.run(args)

    def copy_dir(self, relative):
        peer = self.nodes[1]
        dest = str(Path(peer["root"]) / relative)
        self.remotes[1].call("run", argv=["mkdir", "-p", dest])
        sources.run([*RSYNC, "--exclude=.git", "--exclude=.zig-cache", "--exclude=zig-out",
                     "--exclude=__pycache__", "-e", shlex.join(SSH),
                     str(self.root / relative) + "/", peer["target"] + ":" + dest + "/"])

    def download(self, verify_weights=False):
        target = self.transfer_target()
        self.sync_code()
        head = self.remotes[0].call("download", node=self.nodes[0], recipe=self.recipe,
                                    hf_version=self.lock["huggingface_hub"], verify_weights=verify_weights)
        entries = cache_sync.manifest(self.nodes[0]["hub_cache"], self.recipe, head, verify_weights=verify_weights)
        peer = self.nodes[1]
        self.remotes[1].call("cache-prepare", cache=peer["hub_cache"], entries=entries)
        print("Syncing weights to worker" + (" with checksum scans..." if verify_weights else " (size checks only)..."), flush=True)
        # Explicit NUL-delimited paths preserve snapshot links and copy only their required blobs.
        # No HF_HOME, credentials, other revisions, deletion, or in-place blob writes.
        with tempfile.TemporaryFile() as listing:
            listing.write(b"".join(e["path"].encode() + b"\0" for e in entries))
            listing.seek(0)
            sources.run([*RSYNC, "--checksum" if verify_weights else "--size-only", "--partial-dir=.rsync-partial",
                         "--no-whole-file", "--from0", "--files-from=-", "-e", shlex.join(SSH),
                         self.nodes[0]["hub_cache"] + "/", target + ":" + peer["hub_cache"] + "/"],
                        stdin=listing)
        print("Checking worker " + ("SHA256 hashes..." if verify_weights else "files and sizes..."), flush=True)
        worker = self.remotes[1].call("cache-verify", cache=peer["hub_cache"], entries=entries, receipt=head,
                                     verify_weights=verify_weights)
        return [head, worker]

    def verify_weights(self):
        receipt = read_json(self.root / ".stack/download.json")
        if any(receipt.get(key) != self.recipe[key] for key in ("model", "engram")):
            raise StackError("Download receipt belongs to another checkpoint; select its recipe")
        entries = cache_sync.manifest(self.nodes[0]["hub_cache"], self.recipe, receipt, verify_weights=True)
        return self.pair("verify-weights", [{"cache": n["hub_cache"], "entries": entries} for n in self.nodes])

    def transfer_target(self):
        """Fail closed rather than moving large artifacts over an unverified management route."""
        info = self.doctor()  # Both identities, current fabric, resources, and owned containers.
        if any(s.get("running") or s["state"] == "restarting" for s in info["containers"]):
            raise StackError("Stop this stack before syncing weights or images")
        net = self.config["network"]
        for rank in (0, 1):
            self.remotes[rank].call("route", routes=[(net["nodes"][1-rank]["addresses"][0],
                                                      net["nodes"][rank]["nic"])])
        target = ssh_target(info["probes"][1]["username"] + "@" + net["nodes"][1]["addresses"][0])
        peer = Remote(self.root, {**self.nodes[1], "target": target}).call("probe")
        if peer["identity"] != self.nodes[1]["identity"]:
            raise StackError("Fabric SSH destination identifies a different worker")
        print(f"Bulk transfers: {target} over {net['nodes'][0]['nic']} (SSH/TCP on the RoCE fabric)", flush=True)
        return target

    def sync_image(self, image, target):
        local = self.remotes[0].call("image", image=image)
        # Docker archives do not preserve RepoDigests. Resolve digest references on the head,
        # then inspect/run the immutable image ID on the worker without another registry pull.
        reference = local["id"] if "@sha256:" in image else image
        peer = self.remotes[1].call("image", image=reference, missing_ok=True)
        if peer is None or peer["id"] != local["id"]:
            saved = local["id"] if "@sha256:" in image else image
            try:
                processes.pipeline(["docker", "save", saved], [*SSH, target, "docker load"])
            except subprocess.CalledProcessError as exc:
                raise StackError("Image transfer failed") from exc
        peer = self.remotes[1].call("image", image=reference)
        if peer["id"] != local["id"]:
            raise StackError("Image IDs differ after transfer")
        return local["id"]

    def build(self):
        target = self.transfer_target()
        context = sources.build_context(self.root)
        rid, image = self.plan["runtime"], self.plan["image"]
        dockerfile = str(self.root / "docker/Dockerfile")
        common = ["docker", "build", "--progress=plain", "-f", dockerfile, "--build-arg", f"ZIG_COMMIT={rid}"]
        sources.run([*common, "--target", "build", "-t", image + "-builder", str(context)])
        sources.run([*common, "--label", f"io.spark-tensorfold.runtime={rid}", "-t", image, str(context)])
        sources.run(["bash", str(self.root / "scripts/check-image.sh"), image])
        self.sync_code()
        image_id = self.sync_image(image, target)
        self.remotes[1].call("run", argv=["bash", str(Path(self.nodes[1]["root"]) / "scripts/check-image.sh"), image])
        atomic_json(self.root / ".stack/image.json", {"runtime": rid, "image": image, "id": image_id})
        return {"image": image, "id": image_id, "both_ranks": True}

    def prepare(self):
        if any(s.get("running") or s["state"] == "restarting" for s in self.states()):
            raise StackError("Stop this stack before preparing assets; mounted assets must not change during serving")
        target = self.transfer_target()
        sources.verify(self.root)
        self.sync_code()
        for relative in (".stack/source", ".stack/python-source"):
            self.copy_dir(relative)
        images = self.pair("image", [{"image": self.plan["image"]}] * 2)
        if images[0]["id"] != images[1]["id"]: raise StackError("Build and copy the same runtime image first")
        self.remotes[0].call("run", argv=["docker", "pull", self.lock["asset_image"]])
        self.asset_image = self.sync_image(self.lock["asset_image"], target)
        # Each host builds its own CPU RoPE/Engram tables. Portable AOT cubins are reused on the worker.
        first = self.remotes[0].call("prepare", **self.request(0))
        self.copy_dir(f".stack/assets/{self.plan['asset_key']}/aot")
        second = self.remotes[1].call("prepare", **self.request(1))
        return [first, second]

    def preflight(self):
        self.doctor()
        self.states()  # Includes ownership and two-node reachability checks before any launch mutation.
        images = self.pair("check", [self.request(i) for i in (0, 1)])
        if images[0]["id"] != images[1]["id"]: raise StackError("Runtime image IDs differ")
        return images[0]["id"]

    def stop(self):
        states = self.states()
        # Disable both restart policies before the head drains and causes the follower to exit.
        for i, state in enumerate(states):
            if state["state"] != "absent":
                self.remotes[i].call("run", argv=["docker", "update", "--restart=no", self.nodes[i]["name"]])
        for i in (0, 1):
            state = self.remotes[i].call("state", name=self.nodes[i]["name"], stack=self.config["id"])
            if state.get("running") or state["state"] == "restarting":
                # The container's stored --stop-timeout belongs to the configuration actually running.
                self.remotes[i].call("run", argv=["docker", "stop", self.nodes[i]["name"]])
        after = self.states()
        if any(s.get("running") or s["state"] == "restarting" for s in after):
            raise StackError("Both ranks could not be confirmed stopped")
        return after

    def wait_ready(self, image_id):
        s = self.recipe["serve"]
        deadline = time.monotonic() + s["ready_timeout"]
        last = "waiting for model loading"
        interactive = sys.stdout.isatty()
        stopped = threading.Event()

        def indicate_wait():
            frames = "|/-\\"
            frame = 0
            while not stopped.is_set():
                print(f"\r[{frames[frame % len(frames)]}] Waiting for API to come online...",
                      end="", flush=True)
                frame += 1
                stopped.wait(0.15)

        indicator = threading.Thread(target=indicate_wait, daemon=True) if interactive else None
        if indicator:
            indicator.start()
        else:
            print("Waiting for API to come online...", flush=True)
        try:
            while time.monotonic() < deadline:
                states = self.states()
                for rank, state in enumerate(states):
                    if (not state.get("running") or state["state"] != "running" or state["image"] != image_id
                            or state["restarts"] or state["oom"]):
                        raise StackError(f"rank {rank} failed readiness: {state}")
                try:
                    models = api(s["port"], "/v1/models")
                    if s["name"] in [m["id"] for m in models.get("data", [])]:
                        break
                    last = "API answered with a different model"
                except (OSError, ValueError) as exc:
                    last = str(exc)
                time.sleep(5)
            else:
                raise StackError(f"Readiness timed out after {s['ready_timeout']} s: {last}")
        finally:
            stopped.set()
            if indicator:
                indicator.join()
                print("\r" + " " * 48 + "\r", end="", flush=True)
        print("API online; verifying model and generation...", flush=True)
        return self.verify(image_id=image_id)

    def start(self):
        expected = self.preflight()
        states = self.states()
        running = [s.get("running", False) or s["state"] == "restarting" for s in states]
        if all(running): return self.verify(image_id=expected)
        if any(running): raise StackError("Only one rank is running; use ./stack restart to drain and recreate the pair")
        requests = [{"ports": [self.config["network"]["port"]] + ([self.recipe["serve"]["port"]] if i == 0 else [])} for i in (0, 1)]
        self.pair("idle", requests)
        for i, state in enumerate(states):
            if state["state"] != "absent": self.remotes[i].call("run", argv=["docker", "rm", self.nodes[i]["name"]])
        created = False
        try:
            for i in (1, 0):
                created = True
                self.remotes[i].call("run", argv=self.nodes[i]["command"])
            result = self.wait_ready(expected)
        except BaseException:
            if created:
                try: self.stop()
                except Exception as exc: print(f"Failed-start cleanup: {exc}", file=sys.stderr)
            raise
        atomic_json(self.root / ".stack/last-launch.json", {"recipe": self.recipe, "cluster": self.config,
                                                            "plan": self.plan, "image_id": expected})
        return result

    def restart(self):
        self.preflight()  # A missing image or stale assets must not stop a healthy pair.
        previous = self.root / ".stack/last-launch.json"
        if previous.exists():
            atomic_json(self.root / ".stack/rollback.json", read_json(previous))
        self.stop()
        try: return self.start()
        except Exception:
            print("Restart failed; stopped containers and caches remain. Use ./stack rollback for the last successful launch.", file=sys.stderr)
            raise

    def verify(self, image_id=None):
        image_id = image_id or self.preflight()
        states = self.states()
        for i, state in enumerate(states):
            labels = state.get("labels", {})
            if (state["state"] != "running" or state.get("image") != image_id or state.get("oom")
                    or state.get("env_mismatches") or not state.get("command_matches", False)
                    or labels.get("io.spark-tensorfold.runtime") != self.plan["runtime"]
                    or labels.get("io.spark-tensorfold.assets") != self.plan["asset_key"]
                    or labels.get("io.spark-tensorfold.config") != digest(self.nodes[i]["env"] | self.recipe["serve"])):
                raise StackError(f"rank {i} does not match the requested runtime and settings; use ./stack restart")
        s = self.recipe["serve"]
        models = api(s["port"], "/v1/models")
        if s["name"] not in [m["id"] for m in models.get("data", [])]: raise StackError("Wrong API model")
        health = api(s["port"], "/health")
        if health.get("fatal") or health.get("stalled"): raise StackError(f"Engine unhealthy: {health}")
        reply = api(s["port"], "/v1/chat/completions", {"model": s["name"], "temperature": 0,
                    "max_tokens": 64, "reasoning_effort": "none", "messages": [
                        {"role": "user", "content": "What is 17 * 23? Reply with only the integer."}]}, timeout=300)
        content = reply["choices"][0]["message"].get("content") or ""
        if not re.search(r"\b391\b", content): raise StackError(f"Generation canary failed: {content!r}")
        return {"model": s["name"], "image_id": image_id, "ranks": states,
                "generation": content, "usage": reply.get("usage"), "health": health}

    def status(self):
        result = {"nodes": []}
        ok = True
        for i, remote in enumerate(self.remotes):
            try: state = remote.call("state", name=self.nodes[i]["name"], stack=self.config["id"])
            except StackError as exc: state = {"state": "unknown", "error": str(exc)}
            result["nodes"].append(state)
            ok &= state["state"] == "running"
        try: result["health"] = api(self.recipe["serve"]["port"], "/health")
        except (OSError, ValueError) as exc: result["api_error"] = str(exc); ok = False
        if result.get("health", {}).get("fatal") or result.get("health", {}).get("stalled"): ok = False
        return result, ok

    def logs(self, rank="both", tail=100):
        for i in ((0, 1) if rank == "both" else (int(rank),)):
            print(f"--- rank {i} ---", flush=True)
            self.remotes[i].call("state", name=self.nodes[i]["name"], stack=self.config["id"])
            self.remotes[i].call("run", argv=["docker", "logs", "--tail", str(tail), self.nodes[i]["name"]])

    def doctor(self):
        probes = self.pair("probe")
        for i, probe in enumerate(probes):
            if probe["identity"] != self.nodes[i]["identity"]:
                raise StackError(f"rank {i} hostname/SSH target now identifies a different machine")
        current = choose_network(*probes)
        if current["master"] != self.config["network"]["master"] or current["nodes"] != self.config["network"]["nodes"]:
            raise StackError("RDMA mapping changed since setup; review cluster.json before operating")
        return {"probes": probes, "containers": self.states(), "network": "matches saved configuration"}
