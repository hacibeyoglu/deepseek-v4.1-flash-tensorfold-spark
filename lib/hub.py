"""Download immutable model snapshots into the node's normal Hugging Face cache."""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys


def resolve_hf_token():
    """Use this node's HF_TOKEN, including exports below .bashrc's interactive guard."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    # Capture both streams: shell banners/xtrace must never expose a credential in logs.
    # NUL framing separates the value from any startup banner. No secret is in argv.
    try:
        result = subprocess.run(["bash", "-ic", "set +x; printf '\\0%s\\0' \"${HF_TOKEN:-}\""],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("Cannot load this node's HF_TOKEN; export it before running ./stack download") from None
    parts = result.stdout.split(b"\0")
    token = parts[1].decode().strip() if result.returncode == 0 and len(parts) >= 3 else ""
    if not token:
        raise RuntimeError("HF_TOKEN is unset on this node; export it in the environment or ~/.bashrc")
    return token


def authenticated_client():
    token = resolve_hf_token()
    os.environ["HF_TOKEN"] = token  # Set before the Hub client reads its environment.
    from huggingface_hub import HfApi
    client = HfApi(token=token)
    try:
        client.whoami(token=token)
    except Exception:
        raise RuntimeError("HF_TOKEN authentication failed on this node; check token validity and Hub connectivity") from None
    print("Hugging Face authentication: HF_TOKEN verified", file=sys.stderr, flush=True)
    return client, token


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb", buffering=0) as stream:
        offset = 0
        while chunk := stream.read(16 << 20):
            result.update(chunk)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), offset, len(chunk), os.POSIX_FADV_DONTNEED)
            offset += len(chunk)
    return result.hexdigest()


def child(root, name):
    # Snapshots intentionally contain symlinks to ../.. /blobs; validate the filename, not its target.
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe checkpoint filename: {name!r}")
    return Path(root) / path


def verify_model(snapshot, *, verify_weights=True):
    snapshot = Path(snapshot)
    for name in ("config.json", "tokenizer.json"):
        json.loads((snapshot / name).read_text())
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        files = set(json.loads(index.read_text())["weight_map"].values())
        for name in files:
            if not child(snapshot, name).is_file():
                raise RuntimeError(f"Missing model shard: {name}")
    manifest = snapshot / "SHA256SUMS"
    if not manifest.is_file():
        raise RuntimeError("The pinned Flash17 pack must include SHA256SUMS")
    checked = []
    for line in manifest.read_text().splitlines():
        if not line.strip(): continue
        wanted, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = child(snapshot, name)
        if not path.is_file():
            raise RuntimeError(f"Missing model shard: {name}")
        if verify_weights and file_hash(path) != wanted:
            raise RuntimeError(f"SHA256 mismatch: {name}")
        checked.append(name)
        if verify_weights:
            print(f"verified {name}", file=sys.stderr, flush=True)
    if not checked:
        raise RuntimeError("Empty SHA256SUMS")
    return checked


def verify_hub_model(snapshot, info, revision, *, verify_weights=True):
    """Check pinned metadata; read tensor payloads only when explicitly requested."""
    if info.sha != revision:
        raise RuntimeError("Repository did not resolve to its pinned commit")
    snapshot = Path(snapshot)
    hashes = {}
    shards = set()
    for item in info.siblings:
        name = item.rfilename
        path = child(snapshot, name)
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"Missing file or size mismatch: {name}")
        lfs = item.lfs
        if name.endswith(".safetensors"):
            shards.add(name)
            if lfs is None:
                raise RuntimeError(f"Missing LFS SHA256 for shard: {name}")
        if lfs is not None:
            wanted = lfs.sha256
            if (not wanted or not re.fullmatch(r"[0-9a-f]{64}", wanted)
                    or (verify_weights and file_hash(path) != wanted)):
                raise RuntimeError(f"Hub SHA256 mismatch or missing hash: {name}")
            hashes[name] = wanted
        else:
            # Small Git files have blob SHA1 identifiers, not content SHA256 identifiers.
            data = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob != item.blob_id:
                raise RuntimeError(f"Hub Git blob mismatch: {name}")
            hashes[name] = hashlib.sha256(data).hexdigest()
        if verify_weights:
            print(f"verified {name}", file=sys.stderr, flush=True)
    required = {"config.json", "tokenizer.json", "model.safetensors.index.json"}
    if not required <= hashes.keys() or not shards:
        raise RuntimeError("Hub metadata lacks model configuration, tokenizer, index or shards")
    for name in required:
        json.loads(child(snapshot, name).read_text())
    indexed = set(json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"].values())
    if not indexed or not indexed <= shards:
        raise RuntimeError("Model index references unverified shards")
    return hashes


def engram_files(index):
    files = sorted({v for k, v in index["weight_map"].items() if ".engram.embed." in k})
    if not files:
        raise RuntimeError("The original checkpoint has no Engram embeddings")
    for name in files: child("/unused", name)
    return files


def download(spec):
    verify_weights = spec.get("verify_weights", False)
    client, token = authenticated_client()
    from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
    common = {"cache_dir": spec["cache"], "max_workers": 4, "token": token}
    model, engram = spec["model"], spec["engram"]
    index_path = hf_hub_download(repo_id=engram["repo"], revision=engram["revision"],
                                 filename="model.safetensors.index.json", cache_dir=spec["cache"], token=token)
    files = engram_files(json.loads(Path(index_path).read_text()))
    infos = [client.model_info(m["repo"], revision=m["revision"], files_metadata=True) for m in (model, engram)]
    required = 0
    for m, info, allow in zip((model, engram), infos, (None, set(files) | {"config.json", "model.safetensors.index.json"})):
        if info.sha != m["revision"]: raise RuntimeError("Repository did not resolve to its pinned commit")
        for file in info.siblings:
            if allow is not None and file.rfilename not in allow: continue
            cached = try_to_load_from_cache(m["repo"], file.rfilename, revision=m["revision"], cache_dir=spec["cache"])
            if not isinstance(cached, str) or not Path(cached).is_file() or Path(cached).stat().st_size != file.size:
                required += file.size or 0
    cache = Path(spec["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache).free < required + (2 << 30):
        raise RuntimeError(f"Hugging Face cache needs {required / (1<<30):.1f} GiB plus 2 GiB download headroom")
    model_path = snapshot_download(repo_id=model["repo"], revision=model["revision"], **common)
    original = snapshot_download(repo_id=engram["repo"], revision=engram["revision"],
                                 allow_patterns=["config.json", "model.safetensors.index.json", *files], **common)
    for info, folder, allow in zip(infos, (model_path, original),
                                  (None, set(files) | {"config.json", "model.safetensors.index.json"})):
        for item in info.siblings:
            if allow is not None and item.rfilename not in allow: continue
            path = child(folder, item.rfilename)
            if not path.is_file() or path.stat().st_size != item.size:
                raise RuntimeError(f"Missing file or size mismatch: {item.rfilename}")
    mode = model.get("verification", "sha256sums")
    if mode == "hub-sha256":
        model_hashes = verify_hub_model(model_path, infos[0], model["revision"], verify_weights=verify_weights)
        checked = model_hashes
    elif mode == "sha256sums":
        checked = verify_model(model_path, verify_weights=verify_weights)
        model_hashes = None
    else:
        raise RuntimeError(f"Unknown model verification mode: {mode}")
    info = infos[1]
    if info.sha != engram["revision"]:
        raise RuntimeError("Engram revision did not resolve to its pinned commit")
    metadata = {f.rfilename: f for f in info.siblings}
    for name in files:
        lfs = metadata[name].lfs
        wanted = lfs.sha256 if lfs is not None else None
        if not wanted or (verify_weights and file_hash(child(original, name)) != wanted):
            raise RuntimeError(f"Engram SHA256 mismatch or missing LFS hash: {name}")
        if verify_weights:
            print(f"verified original Engram shard {name}", file=sys.stderr, flush=True)
    receipt = {"model": model, "engram": engram, "weights_verified": verify_weights,
               "model_files_verified": len(checked) if verify_weights else 0, "engram_files": files,
               "engram_sha256": {name: metadata[name].lfs.sha256 for name in files}}
    if model_hashes is not None:
        receipt["model_sha256"] = model_hashes
    return receipt


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--check-auth"]:
            authenticated_client()
            result = {"authenticated": True, "credential": "HF_TOKEN"}
        else:
            result = download(json.load(sys.stdin))
        print(json.dumps(result))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
