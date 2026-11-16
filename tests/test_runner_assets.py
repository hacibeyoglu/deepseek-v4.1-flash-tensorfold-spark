"""Offline runner asset integrity and opt-in profile contracts."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_runner_assets as checker
import fetch_runner_assets as fetcher
from lib.config import load_recipe, plan, runtime_id, synthetic_cluster
from lib import sources
from scripts import stack


class RunnerAssetTests(unittest.TestCase):
    def fixture(self, root, files, extra=None):
        manifest = root / "assets.sha256"
        manifest.write_text("".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items())))
        raw = io.BytesIO()
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
            if extra is not None: tar.addfile(extra)
        data = raw.getvalue()
        archive = root / "assets.tgz"
        archive.write_bytes(data)
        release = root / "assets.release"
        release.write_text(f"url https://example.invalid/assets.tgz\nsha256 {hashlib.sha256(data).hexdigest()}\n")
        return manifest, release, archive

    def test_offline_archive_checks_all_files_and_names_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, release, archive = self.fixture(root, {"calc.tfpc": b"circuit", "bank/bank.json": b"{}"})
            dest = root / "runner"
            with patch.object(fetcher, "download", side_effect=AssertionError("offline")):
                fetcher.fetch(dest, release, manifest, archive)
            self.assertEqual(checker.check(manifest, dest), [])
            (dest / "calc.tfpc").write_bytes(b"changed")
            (dest / "bank/bank.json").unlink()
            (dest / "extra").write_bytes(b"x")
            self.assertEqual(checker.check(manifest, dest), ["missing bank/bank.json", "unlisted extra", "sha256 differs calc.tfpc"])

    def test_bad_archive_hash_fails_before_extracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, release, archive = self.fixture(root, {"calc.tfpc": b"circuit"})
            archive.write_bytes(archive.read_bytes() + b"changed")
            with self.assertRaisesRegex(fetcher.AssetError, "sha256 mismatch"):
                fetcher.fetch(root / "runner", release, manifest, archive)
            self.assertEqual(list((root / "runner").iterdir()), [])

    def test_escaping_names_and_archive_links_are_rejected(self):
        for name, kind in (("../outside", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entry = tarfile.TarInfo(name)
                entry.type, entry.linkname = kind, "/etc/passwd"
                manifest, release, archive = self.fixture(root, {"calc.tfpc": b"circuit"}, entry)
                with self.assertRaisesRegex(fetcher.AssetError, "unsafe entry"):
                    fetcher.fetch(root / "runner", release, manifest, archive)
                self.assertEqual(list((root / "runner").iterdir()), [])

    def test_unlisted_links_and_bad_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, release, archive = self.fixture(root, {"calc.tfpc": b"circuit"})
            fetcher.fetch(root / "runner", release, manifest, archive)
            (root / "runner/link").symlink_to("calc.tfpc")
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                checker.check(manifest, root / "runner")
            manifest.write_text(f"{'0' * 64}  ../outside\n")
            with self.assertRaisesRegex(ValueError, "bad manifest line"):
                checker.listed(manifest)

    def test_downloaded_archive_is_checked_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, release, archive = self.fixture(root, {"calc.tfpc": b"circuit"})
            def download(url, dest):
                shutil.copyfile(archive, dest)
            with patch.object(fetcher, "download", side_effect=download) as fetch:
                fetcher.fetch(root / "runner", release, manifest)
            fetch.assert_called_once()
            self.assertEqual(list(root.glob(".runner-*")), [])
            self.assertEqual(checker.check(manifest, root / "runner"), [])

    def test_runtime_and_build_context_track_runner_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = json.loads((ROOT / "sources.lock.json").read_text())
            (root / "sources.lock.json").write_text(json.dumps(lock))
            names = ["docker/Dockerfile", *lock["runner_assets"].values(), "scripts/fetch_runner_assets.py", "scripts/check_runner_assets.py"]
            for name in names:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            for name in lock["engine"].get("patches", {}):
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            (root / ".stack/engine").mkdir(parents=True)
            (root / ".stack/engine/native.zig").write_text("engine")
            with patch.object(sources, "verify"):
                context = sources.build_context(root)
            self.assertEqual((context / "src/native.zig").read_text(), "engine")
            for name in names[1:]:
                self.assertEqual((context / "runner-tooling" / name).read_bytes(), (ROOT / name).read_bytes())
                before = runtime_id(root)
                with (root / name).open("a") as stream: stream.write("\n# changed\n")
                self.assertNotEqual(before, runtime_id(root))

    def test_compute_is_opt_in_and_preserves_checkpoint_and_capacity(self):
        baseline = load_recipe(ROOT, "flash17-jay-exl3-2.8bpw")
        compute = load_recipe(ROOT, "flash17-jay-exl3-2.8bpw-compute")
        for key in ("model", "engram", "serve"):
            self.assertEqual(baseline[key], compute[key])
        for recipe in ROOT.joinpath("recipes").glob("*.json"):
            env = load_recipe(ROOT, str(recipe))["env"]
            if recipe.stem.endswith("-compute"): continue
            for key in ("TF_DSV41_COMPUTE_LANE", "TF_DSV41_COMPUTE_LANE_DEFAULT", "TF_DSV41_COMPUTE_LANE_RUNNER", "TF_DSV41_RUNNER"):
                self.assertEqual(env[key], "0")
        self.assertEqual(compute["env"]["TF_DSV41_COMPUTE_LANE_DEFAULT"], "1")
        cluster = synthetic_cluster(ROOT, "192.0.2.11,192.0.2.12")
        before, after = plan(ROOT, baseline, cluster), plan(ROOT, compute, cluster)
        self.assertNotEqual(before["asset_key"], after["asset_key"])
        for rank in after["nodes"]:
            self.assertEqual(rank["env"]["TF_DSV41_RUNNER"], "1")
            self.assertNotIn("TF_CUDA_SCHED", rank["env"])
        for key in baseline["env"]:
            if key.startswith(("TF_DSV41_COMPUTE_LANE", "TF_DSV41_RUNNER")) or key == "TF_CUDA_SCHED": continue
            self.assertEqual(baseline["env"][key], compute["env"][key])

    def test_runtime_host_command_uses_native_engine_not_legacy_generators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".stack/engine").mkdir(parents=True)
            (root / ".stack/engine/native.zig").write_text("native")
            (root / ".stack/test-cache/source").mkdir(parents=True)
            (root / ".stack/test-cache/source/stale.zig").write_text("stale")
            with patch.object(stack, "ROOT", root), patch.object(sources, "verify"), patch.object(sources, "run") as run:
                stack.runtime_tests("caller-owned-builder")
            copied = root / ".stack/test-cache/source"
            self.assertEqual((copied / "native.zig").read_text(), "native")
            self.assertFalse((copied / "stale.zig").exists())
            argv = run.call_args.args[0]
            self.assertEqual(argv[argv.index("--network") + 1], "none")
            self.assertIn("test-golden", argv)
            self.assertIn("test", argv)
            self.assertIn("test-dsv41-device-host", argv)
            self.assertIn("test-runner-host", argv)
            self.assertIn("test-runner-frontend-host", argv)


if __name__ == "__main__":
    unittest.main()
