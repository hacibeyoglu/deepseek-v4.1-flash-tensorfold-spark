"""Reproducible source exports and build contexts."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from .config import StackError, atomic_json, read_json, runtime_id


def run(args, **kwargs):
    print("+ " + " ".join(str(x) for x in args), flush=True)
    subprocess.run(args, check=True, **kwargs)


def tree_hash(root):
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if any(x in (".git", "__pycache__", ".zig-cache", "zig-out") for x in p.relative_to(root).parts): continue
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def export(root, spec, destination, cache=None):
    git = Path(cache) if cache else root / ".stack/git" / spec["commit"]
    if cache is None:
        if not git.exists():
            git.mkdir(parents=True)
            run(["git", "init", "--bare", str(git)])
        exists = subprocess.run(["git", "-C", str(git), "cat-file", "-e", spec["commit"] + "^{commit}"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not exists:
            run(["git", "-C", str(git), "fetch", "--depth=1", spec["url"], spec["commit"]])
    archive = subprocess.check_output(["git", "-C", str(git), "archive", "--format=tar", spec["commit"]])
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        # Pinned Git archives contain no untrusted filesystem paths outside the export.
        for member in tf.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise StackError("Unsafe source archive path")
        tf.extractall(destination, filter="data")


def fetch(root, upstream_cache=None, python_cache=None):
    root = Path(root)
    lock = read_json(root / "sources.lock.json")
    manifest = root / ".stack/source.json"
    rid = runtime_id(root)
    if manifest.exists():
        old = read_json(manifest)
        for name in ("source", "python-source"):
            if tree_hash(root / ".stack" / name) != old["hashes"][name]:
                raise StackError(f"Managed {name} has local changes; preserve them before fetching again")
        if old["runtime"] == rid:
            print("Pinned runtime and Python reference exports already match.")
            return
    for name in ("source", "python-source"):
        if (root / ".stack" / name).exists() and not manifest.exists():
            raise StackError(f"Unrecognized .stack/{name}; refusing to overwrite it")
    with tempfile.TemporaryDirectory(prefix="fetch-", dir=root / ".stack") as tmp:
        tmp = Path(tmp)
        export(root, lock["upstream"], tmp / "source", upstream_cache)
        for name in lock["patches"]:
            # Export directories may sit inside the orchestration repository. patch does not discover a parent .git.
            run(["patch", "--batch", "--forward", "--fuzz=0", "--dry-run", "-p1", "-i", str(root / name)], cwd=tmp / "source")
            run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(root / name)], cwd=tmp / "source")
        export(root, lock["python_reference"], tmp / "python-source", python_cache)
        for patch in sorted((tmp / "source/patches").glob("*.patch")):
            run(["patch", "--batch", "-p1", "-i", str(patch)], cwd=tmp / "python-source")
        hashes = {name: tree_hash(tmp / name) for name in ("source", "python-source")}
        for name in hashes:
            dest = root / ".stack" / name
            if dest.exists():
                previous = dest.with_name(name + ".previous")
                if previous.exists(): shutil.rmtree(previous)
                dest.rename(previous)
            (tmp / name).rename(dest)
        atomic_json(manifest, {"runtime": rid, "hashes": hashes, "sources": lock})


def verify(root):
    root = Path(root)
    stamp = read_json(root / ".stack/source.json")
    if stamp["runtime"] != runtime_id(root): raise StackError("Source lock/patches changed; run ./stack fetch")
    for name, expected in stamp["hashes"].items():
        if tree_hash(root / ".stack" / name) != expected:
            raise StackError(f"Source export {name} changed; build only from the reviewed patch set")


def build_context(root):
    root = Path(root)
    verify(root)
    context = root / ".stack/build"
    context.mkdir(exist_ok=True)
    src = context / "src"
    if src.exists(): shutil.rmtree(src)
    shutil.copytree(root / ".stack/source/engine/zig", src,
                    ignore=shutil.ignore_patterns(".zig-cache", "zig-out", "__pycache__"))
    (context / "fatbins").mkdir(exist_ok=True)
    return context
