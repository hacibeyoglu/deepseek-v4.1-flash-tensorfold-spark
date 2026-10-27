"""Transfer only pinned snapshots and their blobs; verify peers without Hub access."""
from __future__ import annotations
import os
import re
from pathlib import Path
import shutil

from .hub import child, file_hash
from .node import snapshot


def manifest(cache, recipe, receipt, *, verify_weights=False):
    root = Path(cache).resolve()
    entries = {}

    def add(path, known_hash=None):
        relative = str(path.relative_to(root))
        if relative in entries:
            return
        if not path.resolve(strict=True).is_relative_to(root):
            raise RuntimeError(f"Cache file escapes transferable cache: {relative}")
        if path.is_symlink():
            target = os.readlink(path)
            resolved = path.resolve(strict=True)
            if os.path.isabs(target) or not resolved.is_relative_to(root):
                raise RuntimeError(f"Cache link escapes transferable cache: {relative}")
            # Preserve relative links, including older hub-level shared blob layouts.
            entries[relative] = {"path": relative, "link": target}
            add(Path(os.path.normpath(path.parent / target)), known_hash)
        elif path.is_file():
            entries[relative] = {"path": relative, "size": path.stat().st_size,
                                 "sha256": known_hash or (file_hash(path) if verify_weights else None)}
        else:
            raise RuntimeError(f"Missing cache file: {relative}")

    for key in ("model", "engram"):
        folder = snapshot(root, recipe[key])
        if key == "model":
            if recipe[key].get("verification", "sha256sums") == "hub-sha256":
                hashes = receipt.get("model_sha256", {})
                if (receipt.get("model") != recipe[key] or not hashes
                        or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values())):
                    raise RuntimeError("Missing or mismatched Hub model checksum receipt")
                files = list(hashes)
            else:
                hashes = {}
                for line in (folder / "SHA256SUMS").read_text().splitlines():
                    if line.strip():
                        value, name = line.split(maxsplit=1)
                        hashes[name.lstrip("*")] = value
                files = [p.relative_to(folder).as_posix() for p in folder.rglob("*")
                         if p.is_file() or p.is_symlink()]
        else:
            hashes = receipt["engram_sha256"]
            files = ["config.json", "model.safetensors.index.json", *receipt["engram_files"]]
        for name in files:
            add(child(folder, name), hashes.get(name))
    return sorted(entries.values(), key=lambda e: e["path"])


def paths(cache, entries):
    root = Path(cache).resolve()
    for entry in entries:
        path = child(root, entry["path"])
        if path == root:
            raise RuntimeError("Empty cache transfer path")
        # rsync must never traverse an existing directory symlink on the destination.
        for parent in path.parents:
            if parent == root:
                break
            if parent.is_symlink():
                raise RuntimeError(f"Symlink in cache destination parents: {parent}")
        if "link" in entry:
            target = entry["link"]
            destination = Path(os.path.normpath(path.parent / target))
            if os.path.isabs(target) or not destination.is_relative_to(root):
                raise RuntimeError(f"Unsafe cache link: {entry['path']}")
        elif path.is_symlink():
            raise RuntimeError(f"Expected a regular cache blob: {entry['path']}")
        if path.is_dir():
            raise RuntimeError(f"Cache file is a directory: {entry['path']}")
        yield path, entry


def prepare(cache, entries):
    root = Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    required, largest = 0, 0
    for path, entry in paths(root, entries):
        if "link" in entry:
            continue
        size = entry["size"]
        largest = max(largest, size)
        if not path.is_file() or path.stat().st_size != size:
            required += size
    # Atomic rsync replacement can temporarily need one additional complete blob.
    if shutil.disk_usage(root).free < required + largest + (2 << 30):
        raise RuntimeError("Worker Hub cache lacks space for weights and atomic transfer headroom")
    return {"required_bytes": required, "replacement_headroom_bytes": largest}


def verify(cache, entries, *, verify_weights=True):
    for path, entry in paths(cache, entries):
        if "link" in entry:
            if not path.is_symlink() or os.readlink(path) != entry["link"] or not path.is_file():
                raise RuntimeError(f"Cache link mismatch: {entry['path']}")
        elif not path.is_file() or path.stat().st_size != entry["size"]:
            raise RuntimeError(f"Cache file missing or size mismatch: {entry['path']}")
        elif verify_weights and (not entry.get("sha256") or file_hash(path) != entry["sha256"]):
            raise RuntimeError(f"Cache SHA256 mismatch or missing hash: {entry['path']}")
    return {"cache_entries_checked": len(entries), "weights_verified": verify_weights}
