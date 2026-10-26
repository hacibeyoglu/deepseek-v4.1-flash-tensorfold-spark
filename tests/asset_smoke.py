#!/usr/bin/env python3
"""CPU-only preparation smoke: synthetic Engram files and real Triton signature imports. No model weights."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import triton

root = Path("/stack")
source = root / ".stack/source"
reference = root / ".stack/python-source/src"
assert triton.__version__ == "3.7.1", triton.__version__
ptxas = subprocess.check_output(["/usr/local/cuda/bin/ptxas", "--version"], text=True)
assert "V13.3.73" in ptxas, ptxas

spec = importlib.util.spec_from_file_location("pack_engram", source / "scripts/pack_engram.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
cfg = {"engram_layer_ids": [1, 14], "engram_max_ngram_size": 3, "engram_vocab_size": 11, "engram_n_heads": 2}

with tempfile.TemporaryDirectory(dir="/out", prefix="fixture-") as tmp:
    tmp = Path(tmp)
    metadata, payload, weights = {}, bytearray(), {}
    for layer, primes in pack.layout(cfg).items():
        rows = sum(primes)
        for part, width in (("weight", 256), ("scale", 8)):
            name = f"layers.{layer}.engram.embed.{part}"
            raw = ((np.arange(rows * width, dtype=np.uint32) + layer) % 256).astype(np.uint8).tobytes()
            metadata[name] = {"dtype": "U8", "shape": [rows, width], "data_offsets": [len(payload), len(payload) + len(raw)]}
            payload.extend(raw)
            weights[name] = "tables.safetensors"
    header = json.dumps(metadata).encode()
    (tmp / "tables.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    packed = tmp / "packed"
    pack.pack(tmp, cfg, packed, [0, 1], 2)
    assert pack.check(tmp, cfg, packed, [0, 1], 2, 16) == 0
    bad = packed / "engram-l1-r0of2.bin"
    with bad.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)[0]
        stream.seek(-1, 2)
        stream.write(bytes([value ^ 1]))
    print("Expected negative check: detect the deliberately corrupted final row.", flush=True)
    assert pack.check(tmp, cfg, packed, [0, 1], 2, 16) == 1

out = Path("/out/sigs.json")
subprocess.run([sys.executable, str(source / "engine/zig/tools/zig/triton_fill.py"), "sigs",
                "--py", str(reference), "--out", str(out)], check=True)
sigs = json.loads(out.read_text())["kernels"]
assert sigs, "No Triton kernels imported"
result = {"synthetic_engram": "both ranks byte checks passed; corruption rejected",
          "triton": triton.__version__, "ptxas": "13.3.73", "kernel_signatures": len(sigs),
          "scope": "CPU fixture/import tests only; not full model preparation or GPU validation"}
Path("/out/result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
