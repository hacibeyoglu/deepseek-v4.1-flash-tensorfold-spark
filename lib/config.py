"""Pure configuration and command generation; no discovery or filesystem mutations."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import copy

RUNTIME_IMAGE = "spark-ds41-tensorfold:latest"


class StackError(RuntimeError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise StackError(f"Cannot read {path}: {exc}") from exc


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temp.open("w", encoding="utf-8") as out:
            os.chmod(temp, 0o600)
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def safe_path(value):
    if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in "\n\r\x00,:"):
        raise StackError(f"Expected an absolute Linux path without commas/colons: {value!r}")
    if value == "/":
        raise StackError("The filesystem root is not a stack directory")
    return value


def ssh_target(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.@:-]*", value):
        raise StackError("Worker must be an SSH alias or user@host, not a shell command")
    return value


def validate_recipe(recipe):
    if recipe.get("schema") != 1:
        raise StackError("Unsupported recipe schema")
    for kind in ("model", "engram"):
        model = recipe[kind]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model["repo"]):
            raise StackError(f"Invalid {kind} repository")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise StackError(f"{kind} revision must be an exact 40-character commit")
    if recipe["model"].get("verification", "sha256sums") not in ("sha256sums", "hub-sha256"):
        raise StackError("model.verification must be sha256sums or hub-sha256")
    s = recipe["serve"]
    for key, low, high in (("port", 1024, 65535), ("context", 2048, 1048576),
                           ("slots", 1, 8), ("max_tokens", 1, 1048576),
                           ("ready_timeout", 1, 86400), ("drain_seconds", 1, 3600)):
        if type(s[key]) is not int or not low <= s[key] <= high:
            raise StackError(f"serve.{key} must be an integer in [{low}, {high}]")
    if not re.fullmatch(r"[\w./:-]+", s["name"]):
        raise StackError("Invalid served model name")
    if s["host"] not in ("0.0.0.0", "127.0.0.1"):
        raise StackError("serve.host must be 0.0.0.0 or 127.0.0.1")
    for key, low, high in (("temperature", 0, 100), ("top_p", 0, 1)):
        if key in s:
            value = s[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not low <= value <= high or (key == "top_p" and value == 0)):
                raise StackError(f"serve.{key} must be a finite number in {'(' if key == 'top_p' else '['}{low}, {high}]")
    if "top_k" in s and (type(s["top_k"]) is not int or not 0 <= s["top_k"] <= 4294967295):
        raise StackError("serve.top_k must be an integer in [0, 4294967295]")
    owned = {"TF_DSV41_CONTEXT", "TF_DSV41_SLOTS", "TF_DSV41_DRAIN_S"}
    for key, val in recipe["env"].items():
        if key in owned or not re.fullmatch(r"(?:TF_DSV41_|GLM53_TF_)[A-Z0-9_]+", key):
            raise StackError(f"Unsupported/managed environment key: {key}")
        if not isinstance(val, str) or any(c in val for c in "\n\r\x00"):
            raise StackError(f"Environment value {key} must be a single-line string")
    return recipe


def load_recipe(root, name):
    path = Path(name)
    if not path.is_file():
        path = Path(root) / "recipes" / (name if name.endswith(".json") else name + ".json")
    return validate_recipe(read_json(path))


def runtime_id(root):
    root = Path(root)
    lock = read_json(root / "sources.lock.json")
    files = [*lock["patches"], "docker/Dockerfile"]
    return digest({"lock": lock, "files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}})[:16]


def pin_image(generated, image_id):
    """Save/launch the resolved image even after a mutable tag moves."""
    pinned = copy.deepcopy(generated)
    for node in pinned["nodes"]:
        command = node["command"]
        command[command.index(pinned["image"])] = image_id
    pinned["image"] = image_id
    return pinned


def model_dir(node, model, container=False):
    cache = "/hf/hub" if container else node["hub_cache"]
    return f'{cache}/models--{model["repo"].replace("/", "--")}/snapshots/{model["revision"]}'


def choose_network(head, worker):
    pairs = []
    used_head, used_worker = set(), set()
    for a in sorted(head["rdma"], key=lambda x: (int(ipaddress.ip_address(x["address"])), x["device"])):
        net = ipaddress.ip_interface(f'{a["address"]}/{a["prefix"]}').network
        for b in worker["rdma"]:
            if (a["address"] != b["address"] and ipaddress.ip_address(b["address"]) in net
                    and b["prefix"] == a["prefix"] and a["device"] not in used_head
                    and b["device"] not in used_worker):
                if a["mtu"] != b["mtu"]:
                    raise StackError(f"RDMA MTU mismatch on {net}: {a['mtu']} vs {b['mtu']}")
                pairs.append((a, b))
                used_head.add(a["device"])
                used_worker.add(b["device"])
                break
    if not pairs:
        raise StackError("No matching active IPv4 RoCE-v2 interfaces; check cabling/IPs on both nodes")
    pairs = pairs[:2]  # This runtime supports one or two native RoCE HCAs.
    return {"master": pairs[0][0]["address"], "port": 29571,
            "nodes": [{"nic": pairs[0][r]["interface"],
                       "hcas": ",".join(f"{p[r]['device']}:{p[r]['gid_index']}" for p in pairs),
                       "nccl_hcas": ",".join(p[r]["device"] for p in pairs),
                       "addresses": [p[r]["address"] for p in pairs]} for r in (0, 1)]}


def validate_cluster(cluster):
    if cluster.get("schema") != 1 or len(cluster.get("nodes", [])) != 2:
        raise StackError("Expected a schema-1 two-node cluster.json; run ./stack setup --worker HOST")
    if not re.fullmatch(r"[0-9a-f]{16}", cluster.get("id", "")):
        raise StackError("Invalid stack ID")
    for i, node in enumerate(cluster["nodes"]):
        if node["rank"] != i:
            raise StackError("Nodes must be ordered head, worker")
        for key in ("root", "hub_cache", "hf_home"):
            safe_path(node[key])
        if i:
            ssh_target(node["target"])
    ipaddress.IPv4Address(cluster["network"]["master"])
    if not 1024 <= cluster["network"]["port"] <= 65535:
        raise StackError("Invalid bootstrap port")
    return cluster


def synthetic_cluster(root, nodes):
    names = nodes.split(",")
    if len(names) != 2:
        raise StackError("--nodes needs exactly head,worker")
    for name in names:
        ipaddress.IPv4Address(name)
    return {"schema": 1, "id": "0123456789abcdef", "nodes": [
        {"rank": i, "root": str(root) if i == 0 else str(Path("/home/user") / Path(root).name),
         "target": None if i == 0 else names[i], "hostname": f"test-r{i}", "uid": 1000, "gid": 1000,
         "hf_home": "/home/user/.cache/huggingface", "hub_cache": "/home/user/.cache/huggingface/hub"}
        for i in (0, 1)], "network": {"master": names[0], "port": 29571, "nodes": [
            {"nic": "rdma0", "hcas": "mlx5_0:3", "nccl_hcas": "mlx5_0", "addresses": [names[i]]} for i in (0, 1)]}}


def effective_env(recipe, cluster, rank):
    env = dict(recipe["env"])
    s, net = recipe["serve"], cluster["network"]
    env.update(TF_DSV41_CONTEXT=str(s["context"]), TF_DSV41_SLOTS=str(s["slots"]),
               TF_DSV41_DRAIN_S=str(s["drain_seconds"]), TF_TP_WORLD="2", TF_TP_RANK=str(rank),
               TF_TP_DEVICE="0", TF_TP_MASTER=net["master"], TF_TP_PORT=str(net["port"]),
               TF_COMM_BACKEND="roce", TF_TP_ROCE_HCA=net["nodes"][rank]["hcas"],
               TF_TP_ROCE_HCAS=str(len(net["nodes"][rank]["hcas"].split(","))),
               NCCL_SOCKET_IFNAME=net["nodes"][rank]["nic"], NCCL_IB_HCA=net["nodes"][rank]["nccl_hcas"])
    return env


def plan(root, recipe, cluster):
    rid = runtime_id(root)
    # Network, API port and host paths do not change asset or session compatibility.
    asset_key = digest({"runtime": rid, "model": recipe["model"], "engram": recipe["engram"],
                        "env": recipe["env"], "context": recipe["serve"]["context"],
                        "slots": recipe["serve"]["slots"]})[:16]
    image = RUNTIME_IMAGE
    result = {"runtime": rid, "asset_key": asset_key, "image": image, "recipe": recipe, "nodes": []}
    for rank, node in enumerate(cluster["nodes"]):
        data = str(Path(node["root"]) / ".stack")
        paths = {"assets": f"{data}/assets/{asset_key}",
                 "engram": f"{data}/engram/{recipe['engram']['revision']}/rank{rank}",
                 # The native cache already keys format, builder sources, shard metadata,
                 # TP layout and DSpark. Unrelated runtime changes must not discard it.
                 "prepared": f"{data}/prepared/{recipe['model']['revision']}",
                 "state": f"{data}/state/{asset_key}", "cache": f"{data}/cache/{asset_key}"}
        name = f"spark-tf-{cluster['id'][:8]}-r{rank}"
        env = effective_env(recipe, cluster, rank)
        env["TF_DSV41_BIAS_VL"] = model_dir(node, recipe["model"], True)
        if env.get("TF_DSV41_NEEDS_MODEL") == "/model":
            env["TF_DSV41_NEEDS_MODEL"] = model_dir(node, recipe["model"], True)
        args = ["docker", "run", "--detach", "--init", "--pull=never", "--name", name,
                "--restart", "unless-stopped", "--stop-timeout", str(recipe["serve"]["drain_seconds"] + 60),
                "--gpus", "all", "--ipc=host", "--network", "host", "--device", "/dev/infiniband",
                "--ulimit", "memlock=-1", "--cap-add", "IPC_LOCK", "--user", f"{node['uid']}:{node['gid']}",
                "--label", f"io.spark-tensorfold.stack={cluster['id']}",
                "--label", f"io.spark-tensorfold.runtime={rid}",
                "--label", f"io.spark-tensorfold.assets={asset_key}", "--label", f"io.spark-tensorfold.rank={rank}",
                "--label", f"io.spark-tensorfold.config={digest(env | recipe['serve'])}"]
        mounts = [(node["hub_cache"], "/hf/hub", True), (paths["assets"], "/assets", True),
                  (paths["engram"], "/engram", True), (paths["prepared"], "/prepared", False),
                  (paths["state"], "/state", False), (paths["state"] + "/sessions", "/sessions", False),
                  (paths["cache"], "/cache", False)]
        for src, dst, readonly in mounts:
            args += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if readonly else "")]
        for key, value in sorted(env.items()):
            args += ["--env", f"{key}={value}"]
        args.append(image)
        model = model_dir(node, recipe["model"], True)
        if rank == 0:
            s = recipe["serve"]
            args += ["tensorfold-dsv41", "serve", model, "--host", s["host"], "--port", str(s["port"]),
                     "--parallel", str(s["slots"]), "--context", str(s["context"]),
                     "--name", s["name"], "--max-tokens", str(s["max_tokens"])]
            for key in ("temperature", "top_p", "top_k"):
                if key in s:
                    args += ["--" + key.replace("_", "-"), str(s[key])]
            if env.get("TF_DSV41_THINKING") == "0": args.append("--no-thinking")
            if env.get("TF_DSV41_DRAFTS", "0") == "0": args.append("--no-drafts")
        else:
            args += ["tf-dsv41-m1", "follow", model, "/assets"]
        result["nodes"].append({**node, "name": name, "env": env, "paths": paths, "command": args})
    return result
