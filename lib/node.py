"""Small stdlib-only worker agent. Its JSON request comes over stdin, never a sourced shell file."""
from __future__ import annotations
import hashlib
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import platform
import pwd
import shutil
import socket
import subprocess
import sys
import threading

try:
    from . import processes
except ImportError:  # Standalone script or embedded SSH agent, before checkout sync.
    import processes


def run(argv, *, cwd=None, env=None, capture=False, input=None, check=True, timeout=None):
    result = processes.run(argv, cwd=cwd, env=None if env is None else {**os.environ, **env},
                            input=input, text=True, stdout=subprocess.PIPE if capture else sys.stderr,
                            stderr=subprocess.PIPE if capture else None, timeout=timeout)
    if check and result.returncode:
        detail = result.stderr.strip() if capture else "see command output above"
        raise RuntimeError(f"{Path(argv[0]).name} exited {result.returncode}: {detail}")
    return result


def cache_paths():
    hf = os.path.expandvars(os.path.expanduser(os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"))))
    hub = os.path.expandvars(os.path.expanduser(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", hf + "/hub"))))
    return os.path.abspath(hf), os.path.abspath(hub)


def node_identity(machine_id, gpu_uuid):
    # Factory/cloned OS images can share /etc/machine-id across distinct Sparks.
    gpu_uuid = gpu_uuid.strip()
    if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
        raise RuntimeError("Expected one stable GPU UUID for this single-GPU stack")
    return hashlib.sha256(machine_id.strip() + b"\0" + gpu_uuid.encode()).hexdigest()


def probe():
    hf, hub = cache_paths()
    addresses = json.loads(run(["ip", "-j", "-4", "addr", "show"], capture=True).stdout)
    interfaces = {a["ifname"]: a for a in addresses if a["operstate"] == "UP"}
    rdma = []
    for device in sorted(Path("/sys/class/infiniband").glob("*")):
        for port in sorted((device / "ports").glob("*")):
            if not (port / "state").read_text().startswith("4:"): continue
            for gid in sorted((port / "gids").glob("*")):
                try:
                    kind = (port / "gid_attrs/types" / gid.name).read_text().strip()
                    iface = (port / "gid_attrs/ndevs" / gid.name).read_text().strip()
                    addr = ipaddress.IPv6Address(gid.read_text().strip()).ipv4_mapped
                except (OSError, ValueError): continue
                if kind != "RoCE v2" or addr is None or iface not in interfaces: continue
                for info in interfaces[iface]["addr_info"]:
                    if info["local"] == str(addr):
                        rdma.append({"device": device.name, "port": int(port.name), "interface": iface,
                                     "address": str(addr), "prefix": info["prefixlen"], "gid_index": int(gid.name),
                                     "mtu": interfaces[iface]["mtu"]})
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    docker = run(["docker", "info", "--format", "{{.Architecture}}"], capture=True).stdout.strip()
    gpu = run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], capture=True).stdout.strip()
    gpu_uuid = run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture=True).stdout.strip()
    if platform.machine() != "aarch64" or "12.1" not in gpu:
        raise RuntimeError(f"This recipe requires ARM64 GB10 (SM121); got {platform.machine()}, {gpu}")
    for tool in ("git", "rsync", "ssh", "python3", "docker", "patch"):
        if shutil.which(tool) is None: raise RuntimeError(f"Missing prerequisite: {tool}")
    return {"hostname": socket.gethostname(), "identity": node_identity(Path('/etc/machine-id').read_bytes(), gpu_uuid),
            "username": pwd.getpwuid(os.getuid()).pw_name, "home": str(Path.home()), "uid": os.getuid(), "gid": os.getgid(),
            "hf_home": hf, "hub_cache": hub, "arch": docker, "gpu": gpu, "rdma": rdma,
            "available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
            "free_disk_bytes": shutil.disk_usage(Path.home()).free}


def inspect(name, stack, expected_env=None, expected_command=None):
    run(["docker", "info"], capture=True)
    result = run(["docker", "inspect", name], capture=True, check=False)
    if result.returncode:
        if "no such object" in result.stderr.lower() or "no such container" in result.stderr.lower():
            return {"state": "absent"}
        raise RuntimeError(result.stderr.strip())
    obj = json.loads(result.stdout)[0]
    labels = obj["Config"].get("Labels") or {}
    if labels.get("io.spark-tensorfold.stack") != stack:
        raise RuntimeError(f"Container {name} is not owned by this stack")
    actual_env = dict(x.split("=", 1) for x in obj["Config"].get("Env", []) if "=" in x)
    return {"state": obj["State"]["Status"], "running": obj["State"]["Running"],
            "image": obj["Image"], "labels": labels, "restarts": obj["RestartCount"],
            "oom": obj["State"]["OOMKilled"], "exit_code": obj["State"]["ExitCode"],
            "restart_policy": obj["HostConfig"]["RestartPolicy"]["Name"],
            "env_mismatches": [k for k, v in (expected_env or {}).items() if actual_env.get(k) != v],
            "command_matches": expected_command is None or obj["Config"].get("Cmd") == expected_command}


def image_id(image, missing_ok=False):
    run(["docker", "info"], capture=True)
    result = run(["docker", "image", "inspect", image], capture=True, check=False)
    if result.returncode:
        if missing_ok and "no such image" in result.stderr.lower(): return None
        raise RuntimeError(result.stderr.strip())
    obj = json.loads(result.stdout)[0]
    if obj["Architecture"] != "arm64": raise RuntimeError(f"Image {image} is not ARM64")
    return {"id": obj["Id"], "labels": obj["Config"].get("Labels") or {}}


def snapshot(root, model):
    return Path(root) / ('models--' + model['repo'].replace('/', '--')) / 'snapshots' / model['revision']


def make_paths(node):
    for path in node["paths"].values(): Path(path).mkdir(parents=True, exist_ok=True)
    (Path(node["paths"]["state"]) / "sessions").mkdir(exist_ok=True)


def docker_base(node, image, env=None, *, tools=False):
    args = ["docker", "run", "--rm", "--init", "--pull=never", "--network", "none",
            "--user", f"{node['uid']}:{node['gid']}", "--env", "HOME=/tmp",
            "--env", "NVIDIA_VISIBLE_DEVICES=void",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONUNBUFFERED=1"]
    if tools:
        args += ["--cpus", "4", "--memory", "16g", "--env", "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
                 "--env", "TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas",
                 "--env", "PYTHONPATH=/stack/.stack/python-source/src"]
    mounts = [(node["root"], "/stack", True), (node["hub_cache"], "/hf/hub", True)]
    mounts.extend((v, '/' + k, k == 'prepared') for k, v in node["paths"].items())
    for src, dst, ro in mounts:
        args += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if ro else "")]
    for key, value in sorted((env or node["env"]).items()): args += ["--env", f"{key}={value}"]
    return args + ["--entrypoint", "python" if tools else "tf-dsv41-m1", image]


def download(request):
    node, recipe = request["node"], request["recipe"]
    venv = Path(node["root"]) / ".stack/venv"
    python = venv / "bin/python"
    version = request["hf_version"]
    if not python.exists():
        run(["python3", "-m", "venv", str(venv)])
    check = run([str(python), "-c", "import huggingface_hub; print(huggingface_hub.__version__)"], capture=True, check=False)
    if check.returncode or check.stdout.strip() != version:
        run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", f"huggingface_hub=={version}"])
    spec = {"cache": node["hub_cache"], "model": recipe["model"], "engram": recipe["engram"],
            "verify_weights": request.get("verify_weights", False)}
    proc = processes.run([str(python), str(Path(node["root"]) / "lib/hub.py")],
                            env={**os.environ, "HF_HOME": node["hf_home"], "HF_HUB_CACHE": node["hub_cache"]},
                            input=json.dumps(spec), stdout=subprocess.PIPE, text=True)
    if proc.returncode: raise RuntimeError("Checkpoint download/verification failed; see output above")
    verified = json.loads(proc.stdout)
    from lib.config import atomic_json
    atomic_json(Path(node["root"]) / ".stack/download.json", verified)
    return verified


SHARED_ASSETS = ("sigs.json", "needs.json", "checked-needs.json", "rope.bin", "engram-host.bin", "aot/aot.json")


def asset_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shared_hashes(assets):
    manifest = json.loads((assets / "aot/aot.json").read_text())
    names = list(SHARED_ASSETS)
    for kernel in manifest["kernels"]:
        value = kernel["hash"]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise RuntimeError("Invalid AOT cubin hash")
        names.append(f"aot/cubins/{value}.cubin")
    return {name: asset_hash(assets / name) for name in sorted(set(names))}


def verify_shared(assets, shared, request):
    identity = {"runtime": request["runtime"], "asset_key": request["asset_key"]}
    if shared.get("identity") != identity:
        raise RuntimeError("Head assets belong to another runtime/recipe")
    # Enumerate from the received AOT manifest, then compare to the head's receipt
    # carried over the control channel. A missing/changed cubin cannot be accepted.
    hashes = shared_hashes(assets)
    if hashes != shared.get("hashes"):
        raise RuntimeError("Copied assets differ from the head's verified assets")
    return hashes


def prepare(request):
    from lib.config import atomic_json, model_dir
    node, recipe, image = request["node"], request["recipe"], request["image"]
    make_paths(node)
    assets, engram = Path(node["paths"]["assets"]), Path(node["paths"]["engram"])
    shared = request.get("shared")
    if node["rank"] == 1:
        if shared is None:
            raise RuntimeError("Worker preparation requires verified head assets")
        verify_shared(assets, shared, request)
    elif shared is not None:
        raise RuntimeError("Only the worker may consume shared head assets")
    model = model_dir(node, recipe["model"], True)
    original = model_dir(node, recipe["engram"], True)
    if not snapshot(node["hub_cache"], recipe["model"]).is_dir(): raise RuntimeError("Run ./stack download first")
    tool_image = request["asset_image"]
    run(["docker", "image", "inspect", tool_image], capture=True)
    tool = docker_base(node, tool_image, tools=True)
    run(tool + ["-c", "import triton,subprocess; assert triton.__version__=='3.7.1'; s=subprocess.check_output(['/usr/local/cuda/bin/ptxas','--version'],text=True); assert 'V13.3.73' in s, s"])
    prefix = "/stack/.stack/source/engine/zig/tools/zig/"
    py = "/stack/.stack/python-source/src"
    packer = "/stack/.stack/source/scripts/pack_engram.py"
    config = json.loads((snapshot(node["hub_cache"], recipe["model"]) / "config.json").read_text())
    config = config.get("text_config", config)
    files = [engram / f"engram-l{layer}-r{node['rank']}of2.bin" for layer in config["engram_layer_ids"]]
    pack_args = ["--src", original, "--config", model + "/config.json", "--rank", str(node["rank"])]
    if not all(p.is_file() for p in files):
        # Conservative peak space for the two half-table output files, before writing any output.
        if shutil.disk_usage(engram).free < 110 * (1 << 30):
            raise RuntimeError("Engram packing needs at least 110 GiB free in .stack (source stays in HF cache)")
        run(tool + [packer, *pack_args, "--out", "/engram"])
    run(tool + [packer, *pack_args, "--check", "/engram"])
    stamp = assets / "prepared.json"
    wanted = {"runtime": request["runtime"], "asset_key": request["asset_key"], "rank": node["rank"]}
    previous = json.loads(stamp.read_text()) if stamp.exists() else {}
    complete = previous.get("identity") == wanted
    if node["rank"] == 0 and complete:
        for name, expected in previous["hashes"].items():
            if asset_hash(assets / name) != expected:
                raise RuntimeError(f"Cached head asset changed: {name}; restore or regenerate these assets")
    if node["rank"] == 0 and not complete:
        run(tool + [prefix + "triton_fill.py", "sigs", "--py", py, "--out", "/assets/sigs.json"])
        run(docker_base(node, image) + ["aot-needs", "/assets/sigs.json", "/assets/needs.json"])
        fill = [prefix + "triton_fill.py", "compile", "--py", py, "--needs", "/assets/needs.json",
                "--options", prefix + "dsv41_triton_options.json", "--arch", "121", "--out", "/assets/aot"]
        if (assets / "aot/aot.json").exists(): fill += ["--have", "/assets/aot"]
        run(tool + fill)
        run(tool + [prefix + "dsv41_rope_tables.py", "--config", model + "/config.json",
                    "--rows", str(recipe["serve"]["context"] + 2048), "--out", "/assets/rope.bin"])
        run(tool + [prefix + "dsv41_engram_host.py", "--py-src", py, "--config", model + "/config.json",
                    "--tokenizer", model + "/tokenizer.json", "--out", "/assets/engram-host.bin"])
    shutil.copyfile(Path(node["root"]) / ".stack/source/config/pfdense-table.json", Path(node["paths"]["cache"]) / "pfdense-table.json")
    if node["rank"] == 0:
        run(docker_base(node, image) + ["aot-needs", "/assets/sigs.json", "/assets/checked-needs.json", "/assets/aot"])
        hashes = shared_hashes(assets)
        shared = {"identity": {"runtime": request["runtime"], "asset_key": request["asset_key"]}, "hashes": hashes}
    else:
        hashes = verify_shared(assets, shared, request)
    atomic_json(stamp, {"identity": wanted, "hashes": hashes})
    return {"prepared": str(assets), "aot": "coverage passed" if node["rank"] == 0 else "head coverage and copied hashes verified",
            "engram": "header, size and sampled rows passed", "shared": shared}


def check_ready_files(request):
    node = request["node"]
    assets = Path(node["paths"]["assets"])
    marker = json.loads((assets / "prepared.json").read_text())
    if marker["identity"] != {"runtime": request["runtime"], "asset_key": request["asset_key"], "rank": node["rank"]}:
        raise RuntimeError("Prepared assets do not match this runtime/recipe/rank; run ./stack prepare")
    for name, expected in marker["hashes"].items():
        if asset_hash(assets / name) != expected:
            raise RuntimeError(f"Asset hash mismatch: {name}; run ./stack prepare")
    manifest = json.loads((assets / "aot/aot.json").read_text())
    for kernel in manifest["kernels"]:
        if not (assets / "aot/cubins" / (kernel["hash"] + ".cubin")).is_file():
            raise RuntimeError("Missing AOT cubin; run ./stack prepare")
    for key in ("model", "engram"):
        folder = snapshot(node["hub_cache"], request["recipe"][key])
        if not folder.is_dir(): raise RuntimeError(f"Missing {key} HF snapshot; run ./stack download")
        if any(p.is_symlink() and not p.exists() for p in folder.rglob("*")):
            raise RuntimeError(f"Broken symlink in {key} HF snapshot")
    downloaded = json.loads((Path(node["root"]) / ".stack/download.json").read_text())
    for key in ("model", "engram"):
        if downloaded[key] != request["recipe"][key]: raise RuntimeError("Download receipt belongs to another checkpoint")
    cfg = json.loads((snapshot(node["hub_cache"], request["recipe"]["model"]) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    import struct
    for layer in cfg["engram_layer_ids"]:
        packed = Path(node["paths"]["engram"]) / f"engram-l{layer}-r{node['rank']}of2.bin"
        with packed.open("rb") as stream: magic, ly, low, high, total, row = struct.unpack("<6Q", stream.read(48))
        if (magic != 0x31344E4531565344 or ly != layer or row != 264 or high <= low
                or high > total or packed.stat().st_size != 4096 + (high-low)*row):
            raise RuntimeError(f"Invalid packed Engram file: {packed}")
    result = image_id(request["image"])
    if result["labels"].get("io.spark-tensorfold.runtime") != request["runtime"]:
        raise RuntimeError("Runtime image label does not match this source and patch set")
    prepared = Path(node["paths"]["prepared"])
    if not any(prepared.iterdir()) and shutil.disk_usage(prepared).free < 110 * (1 << 30):
        raise RuntimeError("First model load needs approximately 95 GiB of prepared weights; leave 110 GiB free")
    return result


def free_ports(ports):
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try: sock.bind(("0.0.0.0", port))
            except OSError as exc: raise RuntimeError(f"Port {port} is occupied: {exc}") from exc


def main(request):
    root = request.get("root")
    if root: sys.path.insert(0, root)
    action = request["action"]
    if action == "probe": return probe()
    if action == "run":
        result = run(request["argv"], cwd=request.get("cwd"), env=request.get("env"), capture=request.get("capture", False))
        return {"stdout": result.stdout if request.get("capture") else ""}
    if action == "claim":
        from_root = Path(root)
        marker = from_root / ".stack/owner.json"
        if marker.exists():
            if json.loads(marker.read_text())["id"] != request["stack"]:
                raise RuntimeError("Worker directory belongs to a different stack")
        elif from_root.exists() and any(from_root.iterdir()):
            raise RuntimeError("Worker directory is not empty and is not managed by this stack; choose --worker-dir")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"id": request["stack"]}) + "\n")
        return {"root": root, "owned": True}
    if action == "state": return inspect(request["name"], request["stack"], request.get("expected_env"), request.get("expected_command"))
    if action == "image": return image_id(request["image"], request.get("missing_ok", False))
    if action == "download": return download(request)
    if action == "verify-weights":
        from lib import cache_sync
        return cache_sync.verify(request["cache"], request["entries"])
    if action in ("cache-prepare", "cache-verify"):
        from lib import cache_sync
        if action == "cache-prepare":
            result = cache_sync.prepare(request["cache"], request["entries"])
            # An interrupted copy must not leave an older receipt authorizing startup.
            (Path(root) / ".stack/download.json").unlink(missing_ok=True)
            return result
        verified = cache_sync.verify(request["cache"], request["entries"],
                                     verify_weights=request.get("verify_weights", False))
        from lib.config import atomic_json
        receipt = {**request["receipt"], **verified, "source": "head-fabric-sync"}
        atomic_json(Path(root) / ".stack/download.json", receipt)
        return receipt
    if action == "prepare": return prepare(request)
    if action == "check": return check_ready_files(request)
    if action == "idle":
        free_ports(request["ports"])
        active = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture=True).stdout.strip()
        if active: raise RuntimeError(f"GPU compute processes already present: {active}; stop the identified workload first")
        return {"idle": True}
    if action == "route":
        for destination, iface in request["routes"]:
            route = json.loads(run(["ip", "-j", "route", "get", destination], capture=True).stdout)[0]
            if route.get("dev") != iface: raise RuntimeError(f"Route to {destination} uses {route.get('dev')}, expected {iface}")
        return {"routes": "matched"}
    raise RuntimeError(f"Unknown worker action: {action}")


if __name__ == "__main__":
    processes.install_signals()
    done = threading.Event()
    try:
        request = json.loads(sys.stdin.readline())
        if request.get("watch_parent"):
            def watch_parent():
                # EOF means the head/SSH connection is gone. Cancel this agent's owned work.
                while os.read(sys.stdin.fileno(), 1):
                    pass
                if not done.is_set():
                    import signal
                    os.kill(os.getpid(), signal.SIGTERM)
            threading.Thread(target=watch_parent, daemon=True).start()
        lock = None
        if request["action"] in ("download", "prepare", "run", "cache-prepare", "cache-verify"):
            lock_path = Path(request["root"]) / ".stack/node.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = lock_path.open("a")
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("A previous operation is still running on this node") from exc
        result = main(request)
        done.set()
        print(json.dumps({"ok": True, "result": result}))
    except KeyboardInterrupt:
        done.set()
        sys.exit(130)
    except Exception as exc:
        done.set()
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
