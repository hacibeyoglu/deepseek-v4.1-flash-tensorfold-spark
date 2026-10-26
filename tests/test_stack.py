"""No Docker, network, GPUs, model downloads, or real cluster configuration."""
from __future__ import annotations
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.config import (StackError, choose_network, digest, load_recipe, model_dir,
                        plan, safe_path, ssh_target, synthetic_cluster, validate_recipe)
from lib.cluster import Cluster, Remote
from lib import hub, node, sources
from scripts import stack


class EntryPointTests(unittest.TestCase):
    def test_cli_and_wrappers_work_outside_checkout(self):
        with tempfile.TemporaryDirectory() as cwd:
            args = ["--dry-run", "-n", "192.0.2.11,192.0.2.12"]
            direct = subprocess.check_output([str(ROOT / "stack"), "start", *args], cwd=cwd, text=True)
            wrapper = subprocess.check_output([str(ROOT / "scripts/start.sh"), *args], cwd=cwd, text=True)
            self.assertEqual(json.loads(direct), json.loads(wrapper))
            for script in (ROOT / "scripts").glob("*.sh"):
                if script.name in ("test.sh", "check-image.sh"):
                    continue
                with self.subTest(script=script.name):
                    subprocess.run([str(script), "--help"], cwd=cwd, check=True, stdout=subprocess.DEVNULL)
        self.assertFalse(list(ROOT.glob("*.sh")))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(ROOT, "flash17-exl3-2.8bpw")
        self.cluster = synthetic_cluster(ROOT, "192.0.2.11,192.0.2.12")

    def test_huggingface_default_and_overrides(self):
        cases = [({}, ("/home/test/.cache/huggingface", "/home/test/.cache/huggingface/hub")),
                 ({"XDG_CACHE_HOME": "/disk/cache"}, ("/disk/cache/huggingface", "/disk/cache/huggingface/hub")),
                 ({"HF_HOME": "~/hf", "HF_HUB_CACHE": "/disk/hub"}, ("/home/test/hf", "/disk/hub")),
                 ({"HF_HOME": "/hf", "HUGGINGFACE_HUB_CACHE": "/legacy"}, ("/hf", "/legacy"))]
        for env, expected in cases:
            with self.subTest(env=env), patch.dict(os.environ, {"HOME": "/home/test", **env}, clear=True):
                self.assertEqual(node.cache_paths(), expected)

    def test_node_specific_paths_and_interfaces(self):
        self.cluster["nodes"][1]["root"] = "/srv/other stack"
        self.cluster["nodes"][1]["hub_cache"] = "/mnt/hf/hub"
        self.cluster["network"]["nodes"][1].update(nic="other0", hcas="roceB:4", nccl_hcas="roceB")
        generated = plan(ROOT, self.recipe, self.cluster)
        head, worker = generated["nodes"]
        self.assertEqual(worker["env"]["TF_TP_ROCE_HCA"], "roceB:4")
        self.assertEqual(worker["env"]["NCCL_SOCKET_IFNAME"], "other0")
        self.assertNotEqual(head["env"]["TF_TP_ROCE_HCA"], worker["env"]["TF_TP_ROCE_HCA"])
        self.assertIn("type=bind,src=/mnt/hf/hub,dst=/hf/hub,readonly", worker["command"])
        self.assertIn("/srv/other stack/.stack", " ".join(worker["command"]))

    def test_snapshot_symlinks_resolve_inside_entire_cache_mount(self):
        node0 = self.cluster["nodes"][0]
        with tempfile.TemporaryDirectory() as tmp:
            node0["hub_cache"] = tmp
            model = Path(model_dir(node0, self.recipe["model"]))
            model.mkdir(parents=True)
            blob = model.parents[1] / "blobs/abc"
            blob.parent.mkdir()
            blob.write_text("weights")
            (model / "weights.safetensors").symlink_to("../../blobs/abc")
            self.assertEqual((model / "weights.safetensors").read_text(), "weights")
            generated = plan(ROOT, self.recipe, self.cluster)
            self.assertIn("type=bind,src=" + tmp + ",dst=/hf/hub,readonly", generated["nodes"][0]["command"])
            self.assertTrue(model_dir(node0, self.recipe["model"], True).startswith("/hf/hub/models--"))

    def test_artifacts_change_with_shape_not_api_port(self):
        original = plan(ROOT, self.recipe, self.cluster)
        self.recipe["serve"]["port"] = 8100
        moved = plan(ROOT, self.recipe, self.cluster)
        self.assertEqual(original["asset_key"], moved["asset_key"])
        self.assertNotEqual(original["nodes"][0]["command"], moved["nodes"][0]["command"])
        self.recipe["serve"]["context"] = 65536
        self.assertNotEqual(original["asset_key"], plan(ROOT, self.recipe, self.cluster)["asset_key"])

    def test_model_revision_invalidates_assets(self):
        first = plan(ROOT, self.recipe, self.cluster)
        self.recipe["model"]["revision"] = "a" * 40
        self.assertNotEqual(first["asset_key"], plan(ROOT, self.recipe, self.cluster)["asset_key"])

    def test_no_precision_change_or_disabled_memory_guard(self):
        env = self.recipe["env"]
        self.assertEqual(env["TF_DSV41_INDEX_KV"], "fp8")
        self.assertEqual(env["TF_DSV41_BOOT_MEASURED"], "auto")
        self.assertEqual(env["TF_DSV41_FLOOR_HARD_GIB"], "4")
        self.assertEqual(env["TF_DSV41_SESSION_DISK_WRITE_THROUGH"], "1")

    def test_validation_rejects_unpinned_checkpoint(self):
        self.recipe["model"]["revision"] = "main"
        with self.assertRaises(StackError): validate_recipe(self.recipe)

    def test_managed_topology_cannot_be_injected_by_recipe(self):
        self.recipe["env"]["TF_TP_RANK"] = "7"
        with self.assertRaises(StackError): validate_recipe(self.recipe)

    def test_shell_and_docker_mount_injection_rejected(self):
        for target in ("-oProxyCommand=evil", "host; touch bad", "$(id)", "host\nwrong"):
            with self.assertRaises(StackError): ssh_target(target)
        for path in ("/", "relative", "/data,dst=/etc", "/a\nb"):
            with self.assertRaises(StackError): safe_path(path)

    def test_dry_run_never_calls_processes(self):
        with patch("subprocess.run", side_effect=AssertionError("dry run spawned a process")), patch("sys.stdout", new=io.StringIO()) as stdout:
            self.assertEqual(stack.main(["run", "flash17-exl3-2.8bpw", "--setup", "--dry-run", "-n", "192.0.2.11,192.0.2.12"]), 0)
            data = json.loads(stdout.getvalue())
        self.assertEqual(data["steps"], ["fetch", "download", "build", "prepare", "start", "verify"])

    def test_nodes_not_accepted_for_real_operations(self):
        with self.assertRaises(StackError): stack.main(["start", "-n", "192.0.2.11,192.0.2.12"])

    def test_launch_both_native_hcas_and_correct_rank(self):
        for i in (0, 1): self.cluster["network"]["nodes"][i]["hcas"] = f"roce{i}a:3,roce{i}b:4"
        generated = plan(ROOT, self.recipe, self.cluster)
        for i, n in enumerate(generated["nodes"]):
            self.assertEqual(n["env"]["TF_TP_ROCE_HCAS"], "2")
            self.assertEqual(n["env"]["TF_TP_RANK"], str(i))
            self.assertIn("TF_TP_ROCE_HCA=" + n["env"]["TF_TP_ROCE_HCA"], n["command"])
            self.assertIn("--pull=never", n["command"])
            self.assertNotIn("--rm", n["command"])


def interface(device, address, iface="net0", gid=3, mtu=9000):
    return {"device": device, "address": address, "interface": iface, "gid_index": gid,
            "prefix": 24, "port": 1, "mtu": mtu}


class NetworkTests(unittest.TestCase):
    def test_cloned_machine_ids_still_distinguish_physical_nodes(self):
        head = node.node_identity(b"cloned-id\n", "GPU-head\n")
        self.assertNotEqual(head, node.node_identity(b"cloned-id\n", "GPU-worker"))
        self.assertEqual(head, node.node_identity(b"cloned-id", "GPU-head"))
        with self.assertRaises(RuntimeError):
            node.node_identity(b"cloned-id", "")

    def test_pair_by_subnet_not_interface_name_or_device_order(self):
        head = {"rdma": [interface("A", "192.0.2.11", "en0"), interface("B", "198.51.100.11", "en1")]}
        worker = {"rdma": [interface("Y", "198.51.100.12", "eth8", 4), interface("X", "192.0.2.12", "eth9", 5)]}
        result = choose_network(head, worker)
        self.assertEqual(result["master"], "192.0.2.11")
        self.assertEqual(result["nodes"][1]["hcas"], "X:5,Y:4")
        self.assertEqual(result["nodes"][1]["nic"], "eth9")

    def test_no_shared_lane_fails(self):
        with self.assertRaises(StackError): choose_network({"rdma": []}, {"rdma": []})

    def test_mtu_mismatch_fails(self):
        with self.assertRaises(StackError):
            choose_network({"rdma": [interface("A", "192.0.2.11")]}, {"rdma": [interface("B", "192.0.2.12", mtu=1500)]})

    def test_duplicate_gid_entries_do_not_duplicate_hcas(self):
        a, b = interface("A", "192.0.2.11"), interface("B", "192.0.2.12")
        result = choose_network({"rdma": [a, a]}, {"rdma": [b, b]})
        self.assertEqual(result["nodes"][0]["hcas"], "A:3")


class FakeRemote:
    def __init__(self, cluster, rank, events):
        self.cluster, self.rank, self.events = cluster, rank, events
        self.state = {"state": "absent"}
        self.fail_action = None
        self.image = "sha256:same"
    def call(self, action, **kwargs):
        self.events.append((self.rank, action, copy.deepcopy(kwargs)))
        if action == self.fail_action: raise StackError("worker unreachable")
        if action == "state": return copy.deepcopy(self.state)
        if action == "check": return {"id": self.image}
        if action == "idle": return {"idle": True}
        if action == "run":
            cmd = kwargs["argv"]
            if cmd[1] == "run":
                labels = dict(cmd[i+1].split("=", 1) for i, arg in enumerate(cmd) if arg == "--label")
                self.state = {"state": "running", "running": True, "image": self.image,
                              "labels": labels, "restarts": 0, "oom": False, "command_matches": True, "env_mismatches": []}
            elif cmd[1] == "stop": self.state.update(state="exited", running=False)
            elif cmd[1] == "rm": self.state = {"state": "absent"}
            return {}
        raise AssertionError(action)


class DockerStateTests(unittest.TestCase):
    def test_absent_container_case_variants_and_daemon_failure(self):
        for error in ("Error: No such object: test", "error: no such object: test",
                      "Error response from daemon: No such container: test"):
            with self.subTest(error=error), patch.object(node, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, "", ""),
                    subprocess.CompletedProcess([], 1, "", error)]):
                self.assertEqual(node.inspect("test", "owned"), {"state": "absent"})
        with patch.object(node, "run", side_effect=[subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 1, "", "permission denied")]):
            with self.assertRaisesRegex(RuntimeError, "permission denied"):
                node.inspect("test", "owned")

    def test_absent_image_case_variants_and_daemon_failure(self):
        for error in ("Error: No such image: test", "error: no such image: test"):
            with self.subTest(error=error), patch.object(node, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, "", ""),
                    subprocess.CompletedProcess([], 1, "", error)]):
                self.assertIsNone(node.image_id("test", missing_ok=True))
        with patch.object(node, "run", side_effect=[subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 1, "", "permission denied")]):
            with self.assertRaisesRegex(RuntimeError, "permission denied"):
                node.image_id("test", missing_ok=True)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        recipe = load_recipe(ROOT, "flash17-exl3-2.8bpw")
        cluster = synthetic_cluster(ROOT, "192.0.2.11,192.0.2.12")
        self.c = Cluster(ROOT, recipe, cluster)
        self.c.root = Path(self.temp.name)
        self.events = []
        self.c.remotes = [FakeRemote(self.c, i, self.events) for i in (0, 1)]
        self.c.wait_ready = lambda image: {"ready": True}
        self.c.doctor = lambda: {"network": "fixture"}

    def mutations(self): return [e for e in self.events if e[1] == "run"]

    def test_worker_starts_before_head(self):
        self.c.start()
        self.assertEqual([e[0] for e in self.mutations() if e[2]["argv"][1] == "run"], [1, 0])
        self.assertTrue((self.c.root / ".stack/last-launch.json").exists())

    def test_unreachable_worker_never_launches_or_stops_head(self):
        self.c.remotes[1].fail_action = "state"
        with self.assertRaises(StackError): self.c.start()
        self.assertEqual(self.mutations(), [])
        with self.assertRaises(StackError): self.c.stop()
        self.assertEqual(self.mutations(), [])

    def test_partial_pair_requires_restart(self):
        self.c.remotes[1].state = {"state": "running", "running": True}
        with self.assertRaisesRegex(StackError, "Only one rank"): self.c.start()
        self.assertEqual(self.mutations(), [])

    def test_mismatched_images_refuse_before_mutation(self):
        self.c.remotes[1].image = "sha256:other"
        with self.assertRaisesRegex(StackError, "image IDs differ"): self.c.start()
        self.assertEqual(self.mutations(), [])

    def test_missing_assets_does_not_stop_existing_pair_on_restart(self):
        self.c.remotes[1].fail_action = "check"
        with self.assertRaises(StackError): self.c.restart()
        self.assertEqual(self.mutations(), [])

    def test_prepare_refuses_to_mutate_live_assets(self):
        self.c.remotes[0].state = {"state": "running", "running": True}
        with self.assertRaisesRegex(StackError, "Stop this stack"): self.c.prepare()
        self.assertEqual(self.mutations(), [])

    def test_both_restart_policies_disabled_before_head_drain(self):
        self.c.start()
        self.events.clear()
        self.c.stop()
        commands = [(e[0], e[2]["argv"][1]) for e in self.mutations()]
        self.assertEqual(commands, [(0, "update"), (1, "update"), (0, "stop"), (1, "stop")])

    def test_failed_readiness_stops_both_and_does_not_mark_success(self):
        self.c.wait_ready = lambda image: (_ for _ in ()).throw(StackError("bad generation"))
        with self.assertRaisesRegex(StackError, "bad generation"): self.c.start()
        self.assertTrue(all(n.state["state"] == "exited" for n in self.c.remotes))
        self.assertFalse((self.c.root / ".stack/last-launch.json").exists())

    def test_running_pair_with_changed_settings_fails_verification(self):
        self.c.start()
        self.c.remotes[0].state["env_mismatches"] = ["TF_DSV41_CONTEXT"]
        with self.assertRaisesRegex(StackError, "does not match"): self.c.verify(image_id="sha256:same")

    def test_generation_is_required_for_success(self):
        self.c.start()
        replies = [{"data": [{"id": self.c.recipe["serve"]["name"]}]}, {"ok": True},
                   {"choices": [{"message": {"content": "392"}}]}]
        with patch("lib.cluster.api", side_effect=replies):
            with self.assertRaisesRegex(StackError, "canary failed"): self.c.verify(image_id="sha256:same")

    def test_worker_crash_aborts_readiness(self):
        self.c.start()
        self.c.remotes[1].state.update(state="exited", running=False)
        with self.assertRaisesRegex(StackError, "failed readiness"):
            Cluster.wait_ready(self.c, "sha256:same")


class AuthenticationTests(unittest.TestCase):
    def test_exported_hf_token_takes_precedence(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_environment_fixture"}, clear=True), patch.object(hub.subprocess, "run") as run:
            self.assertEqual(hub.resolve_hf_token(), "hf_environment_fixture")
            run.assert_not_called()

    def test_noninteractive_worker_loads_own_interactive_token_privately(self):
        result = subprocess.CompletedProcess([], 0, b"banner\n\0hf_worker_fixture\0", b"private shell diagnostics")
        with patch.dict(os.environ, {}, clear=True), patch.object(hub.subprocess, "run", return_value=result) as run:
            self.assertEqual(hub.resolve_hf_token(), "hf_worker_fixture")
            self.assertTrue(run.call_args.kwargs["capture_output"])
            self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
            self.assertNotIn("hf_worker_fixture", str(run.call_args))

    def test_missing_token_refuses_anonymous_fallback(self):
        result = subprocess.CompletedProcess([], 0, b"\0\0", b"")
        with patch.dict(os.environ, {}, clear=True), patch.object(hub.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN is unset"):
                hub.resolve_hf_token()

    def test_invalid_token_error_does_not_expose_credentials(self):
        client = Mock()
        client.whoami.side_effect = RuntimeError("sensitive upstream message hf_fixture_secret")
        module = SimpleNamespace(HfApi=Mock(return_value=client))
        with patch.dict(os.environ, {"HF_TOKEN": "hf_fixture_secret"}, clear=True), patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN authentication failed") as caught:
                hub.authenticated_client()
            self.assertNotIn("hf_fixture_secret", str(caught.exception))

    def test_all_hub_requests_use_explicit_token_with_implicit_auth_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"layer.engram.embed.weight": "original.safetensors"}}))
            original = root / "original.safetensors"
            original.write_bytes(b"engram fixture")
            checksum = hashlib.sha256(original.read_bytes()).hexdigest()
            model, engram = {"repo": "test/model", "revision": "a"*40}, {"repo": "test/engram", "revision": "b"*40}
            client = Mock()
            client.model_info.side_effect = [SimpleNamespace(sha=model["revision"], siblings=[]),
                SimpleNamespace(sha=engram["revision"], siblings=[SimpleNamespace(rfilename=original.name,
                                size=original.stat().st_size, lfs=SimpleNamespace(sha256=checksum))])]
            module = SimpleNamespace(HfApi=Mock(return_value=client), hf_hub_download=Mock(return_value=str(index)),
                                     snapshot_download=Mock(return_value=str(root)), try_to_load_from_cache=Mock(return_value=str(original)))
            output = io.StringIO()
            with patch.dict(os.environ, {"HF_TOKEN": "hf_explicit_fixture", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}, clear=True), \
                 patch.dict(sys.modules, {"huggingface_hub": module}), patch.object(hub, "verify_model", return_value=["model"]), \
                 patch.object(hub.shutil, "disk_usage", return_value=SimpleNamespace(free=1<<40)), patch("sys.stderr", output):
                hub.download({"cache": str(root), "model": model, "engram": engram})
            module.HfApi.assert_called_once_with(token="hf_explicit_fixture")
            client.whoami.assert_called_once_with(token="hf_explicit_fixture")
            self.assertEqual(module.hf_hub_download.call_args.kwargs["token"], "hf_explicit_fixture")
            self.assertEqual(module.snapshot_download.call_count, 2)
            for call in module.snapshot_download.call_args_list:
                self.assertEqual(call.kwargs["token"], "hf_explicit_fixture")
                self.assertEqual(call.kwargs["cache_dir"], str(root))
            self.assertNotIn("hf_explicit_fixture", output.getvalue())


class FilesTests(unittest.TestCase):
    def test_engram_download_selects_only_embedding_shards(self):
        index = {"weight_map": {"layers.1.engram.embed.weight": "a.safetensors",
                                "layers.1.engram.embed.scale": "a.safetensors",
                                "layers.14.engram.embed.weight": "b.safetensors",
                                "unrelated.weight": "huge.safetensors"}}
        self.assertEqual(hub.engram_files(index), ["a.safetensors", "b.safetensors"])
        with self.assertRaises(ValueError): hub.engram_files({"weight_map": {"x.engram.embed.weight": "../bad"}})

    def test_model_hash_verification_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("config.json", "tokenizer.json"): (root / name).write_text("{}")
            (root / "weight.safetensors").write_bytes(b"model")
            (root / "SHA256SUMS").write_text(hashlib.sha256(b"model").hexdigest() + "  weight.safetensors\n")
            self.assertEqual(hub.verify_model(root), ["weight.safetensors"])
            (root / "weight.safetensors").write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"): hub.verify_model(root)

    def test_owned_worker_directory_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unrelated.txt").write_text("keep")
            with self.assertRaisesRegex(RuntimeError, "not empty"):
                node.main({"action": "claim", "root": tmp, "stack": "a"})
            self.assertEqual((root / "unrelated.txt").read_text(), "keep")

    def test_worker_claim_is_repeatable_but_not_shared_with_other_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = {"action": "claim", "root": tmp, "stack": "a"}
            node.main(request)
            node.main(request)
            with self.assertRaisesRegex(RuntimeError, "different stack"): node.main({**request, "stack": "b"})

    def test_missing_worker_response_is_unknown_not_absent(self):
        remote = Remote(ROOT, {"target": "test-worker", "root": "/example"})
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 255, stdout="")):
            with self.assertRaisesRegex(StackError, "unknown"): remote.call("state", name="test", stack="a")


class SourceTests(unittest.TestCase):
    def test_source_export_inside_parent_git_and_repeat_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("patches", "docker", ".stack"): (root / folder).mkdir()
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            (root / "docker/Dockerfile").write_text("FROM example\n")
            (root / "patches/change.patch").write_text(
                "diff --git a/engine/file b/engine/file\n--- a/engine/file\n+++ b/engine/file\n@@ -1 +1 @@\n-old\n+combined\n")
            (root / "sources.lock.json").write_text(json.dumps({"upstream": {}, "python_reference": {}, "patches": ["patches/change.patch"]}))
            def fake_export(root, spec, destination, cache=None):
                destination.mkdir()
                if destination.name == "source":
                    (destination / "engine").mkdir()
                    (destination / "engine/file").write_text("old\n")
                    (destination / "patches").mkdir()
                else: (destination / "reference.py").write_text("pass\n")
            def quiet_run(args, **kwargs):
                subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kwargs)
            with patch("lib.sources.export", side_effect=fake_export), patch("lib.sources.run", side_effect=quiet_run):
                sources.fetch(root)
                self.assertEqual((root / ".stack/source/engine/file").read_text(), "combined\n")
                sources.verify(root)
                with patch("lib.sources.export", side_effect=AssertionError("unexpected refetch")):
                    sources.fetch(root)
                (root / ".stack/source/engine/file").write_text("user edit\n")
                with self.assertRaisesRegex(StackError, "local changes"): sources.fetch(root)
                self.assertEqual((root / ".stack/source/engine/file").read_text(), "user edit\n")

    def test_runtime_id_tracks_patch_changes(self):
        from lib.config import runtime_id
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docker").mkdir()
            (root / "docker/Dockerfile").write_text("FROM fixture\n")
            (root / "change.patch").write_text("first\n")
            (root / "sources.lock.json").write_text(json.dumps({"patches": ["change.patch"]}))
            before = runtime_id(root)
            (root / "change.patch").write_text("second\n")
            self.assertNotEqual(before, runtime_id(root))


if __name__ == "__main__": unittest.main()
