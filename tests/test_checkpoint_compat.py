"""Pinned Hub verification and recipe isolation, without network or model downloads."""
import copy
import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as Item
import unittest
from unittest.mock import Mock, patch

from lib import cache_sync, hub, node
from lib.config import StackError, load_recipe, model_dir, plan, synthetic_cluster, validate_recipe
from scripts import stack

ROOT = Path(__file__).resolve().parents[1]


class HubChecksumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / 'head'
        self.model = {'repo': 'org/model', 'revision': 'a' * 40, 'verification': 'hub-sha256'}
        self.folder = node.snapshot(self.cache, self.model)
        self.folder.mkdir(parents=True)
        self.data = {'config.json': b'{}', 'tokenizer.json': b'{}',
                     'model.safetensors.index.json': b'{"weight_map":{"x":"weights.safetensors"}}',
                     'weights.safetensors': b'checkpoint', 'README.md': b'model card'}
        siblings = []
        for name, data in self.data.items():
            blob = self.folder.parents[1] / 'blobs' / hashlib.sha256(data).hexdigest()
            blob.parent.mkdir(exist_ok=True)
            blob.write_bytes(data)
            (self.folder / name).symlink_to('../../blobs/' + blob.name)
            siblings.append(Item(rfilename=name, size=len(data),
                                 lfs=Item(sha256=hub.file_hash(blob)) if name.endswith('.safetensors') else None,
                                 blob_id=hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()))
        self.info = Item(sha=self.model['revision'], siblings=siblings)

    def verify(self):
        return hub.verify_hub_model(self.folder, self.info, self.model['revision'])

    def test_explicit_mode_required_and_original_files_unchanged(self):
        with self.assertRaisesRegex(RuntimeError, 'SHA256SUMS'):
            hub.verify_model(self.folder)
        hashes = self.verify()
        self.assertEqual(hashes, {n: hashlib.sha256(d).hexdigest() for n, d in self.data.items()})
        self.assertFalse((self.folder / 'SHA256SUMS').exists())
        for name, data in self.data.items():
            self.assertTrue((self.folder / name).is_symlink())
            self.assertEqual((self.folder / name).read_bytes(), data)

    def test_wrong_revision_missing_hash_and_same_size_corruption_rejected(self):
        self.info.sha = 'b' * 40
        with self.assertRaisesRegex(RuntimeError, 'pinned commit'):
            self.verify()
        self.info.sha = self.model['revision']
        shard = next(i for i in self.info.siblings if i.lfs)
        saved = shard.lfs
        shard.lfs = None
        with self.assertRaisesRegex(RuntimeError, 'Missing LFS'):
            self.verify()
        shard.lfs = saved
        (self.folder / shard.rfilename).write_bytes(b'corruption')
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            self.verify()

    def test_git_metadata_corruption_missing_file_and_traversal_rejected(self):
        (self.folder / 'README.md').write_bytes(b'wrong card')
        with self.assertRaisesRegex(RuntimeError, 'Git blob mismatch'):
            self.verify()
        (self.folder / 'README.md').write_bytes(self.data['README.md'])
        (self.folder / 'weights.safetensors').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing file'):
            self.verify()
        self.info.siblings[0].rfilename = '../escape'
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.verify()

    def test_index_cannot_reference_a_shard_missing_from_metadata(self):
        self.info.siblings = [f for f in self.info.siblings if not f.lfs]
        with self.assertRaisesRegex(RuntimeError, 'metadata lacks'):
            self.verify()
        extra = Item(rfilename='other.safetensors', size=1, lfs=Item(sha256=hashlib.sha256(b'x').hexdigest()))
        (self.folder / extra.rfilename).write_bytes(b'x')
        self.info.siblings.append(extra)
        with self.assertRaisesRegex(RuntimeError, 'unverified shards'):
            self.verify()

    def test_download_records_expected_hashes_without_scanning_weights(self):
        original = {'repo': 'org/engram', 'revision': 'b' * 40}
        folder = node.snapshot(self.cache, original)
        folder.mkdir(parents=True)
        index = folder / 'model.safetensors.index.json'
        index.write_text('{"weight_map":{"x.engram.embed.weight":"embedding.bin"}}')
        (folder / 'embedding.bin').write_bytes(b'engram')
        sha = hub.file_hash(folder / 'embedding.bin')
        client = Mock()
        client.model_info.side_effect = [self.info, Item(sha=original['revision'], siblings=[
            Item(rfilename='embedding.bin', size=6, lfs=Item(sha256=sha))])]
        module = Item(hf_hub_download=Mock(return_value=str(index)),
                      snapshot_download=Mock(side_effect=[str(self.folder), str(folder)]),
                      try_to_load_from_cache=Mock(return_value=None))
        with patch.dict(sys.modules, {'huggingface_hub': module}), \
                patch('lib.hub.authenticated_client', return_value=(client, 'fixture-token')), \
                patch('lib.hub.shutil.disk_usage', return_value=Item(free=1 << 40)), \
                patch('lib.hub.file_hash', side_effect=AssertionError('default download hashed weights')):
            receipt = hub.download({'cache': str(self.cache), 'model': self.model, 'engram': original})
        self.assertFalse(receipt['weights_verified'])
        self.assertEqual(receipt['model_files_verified'], 0)
        self.assertEqual(receipt['model_sha256'], self.verify())
        self.assertEqual(receipt['engram_sha256'], {'embedding.bin': sha})
        for call in module.snapshot_download.call_args_list:
            self.assertEqual(call.kwargs['token'], 'fixture-token')
            self.assertEqual(call.kwargs['cache_dir'], str(self.cache))
        self.assertEqual(module.hf_hub_download.call_args.kwargs['token'], 'fixture-token')

    def test_worker_verifies_hub_receipt_offline_and_excludes_unverified_files(self):
        import subprocess
        original = {'repo': 'org/engram', 'revision': 'b' * 40}
        folder = node.snapshot(self.cache, original)
        folder.mkdir(parents=True)
        for name in ('config.json', 'model.safetensors.index.json', 'embedding.bin'):
            (folder / name).write_bytes(b'{}')
        recipe = {'model': self.model, 'engram': original}
        receipt = {**recipe, 'model_sha256': self.verify(), 'engram_files': ['embedding.bin'],
                   'engram_sha256': {'embedding.bin': hub.file_hash(folder / 'embedding.bin')}}
        (self.folder / 'unverified').write_text('do not copy')
        entries = cache_sync.manifest(self.cache, recipe, receipt, verify_weights=True)
        self.assertFalse(any(e['path'].endswith('unverified') for e in entries))
        worker = Path(self.temp.name) / 'worker'
        cache_sync.prepare(worker, entries)
        subprocess.run(['rsync', '-a', '--from0', '--files-from=-', str(self.cache) + '/', str(worker) + '/'],
                       input=''.join(e['path'] + '\0' for e in entries).encode(), check=True)
        with patch('lib.hub.authenticated_client', side_effect=AssertionError('worker accessed Hub')):
            cache_sync.verify(worker, entries)
        (node.snapshot(worker, self.model) / 'weights.safetensors').write_bytes(b'corruption')
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            cache_sync.verify(worker, entries)
        for bad in ({**receipt, 'model_sha256': {}}, {**receipt, 'model': original}):
            with self.assertRaisesRegex(RuntimeError, 'receipt'):
                cache_sync.manifest(self.cache, recipe, bad)


class RecipeCompatibilityTests(unittest.TestCase):
    def test_setup_and_download_require_explicit_verification_flag(self):
        for action in (['run', 'flash17-exl3-2.8bpw', '--setup'], ['download']):
            for verify in (False, True):
                with patch('scripts.stack.Cluster') as cluster, \
                        patch('scripts.stack.read_json', return_value={}), \
                        patch('scripts.stack.locked', return_value=contextlib.nullcontext()), \
                        patch('scripts.stack.sources.fetch'), contextlib.redirect_stdout(io.StringIO()):
                    cluster.return_value.start.return_value = None
                    cluster.return_value.download.return_value = None
                    stack.main(action + (['--verify-weights'] if verify else []))
                    cluster.return_value.download.assert_called_once_with(verify_weights=verify)
        with self.assertRaisesRegex(StackError, 'requires'):
            stack.main(['run', '--verify-weights'])

    def test_new_recipes_keep_precision_and_get_separate_assets(self):
        base = load_recipe(ROOT, 'flash17-exl3-2.8bpw')
        cluster = synthetic_cluster(ROOT, '192.0.2.11,192.0.2.12')
        old = plan(ROOT, base, cluster)
        assets = {old['asset_key']}
        self.assertNotIn('TF_DSV41_NEEDS_MODEL', old['nodes'][0]['env'])
        self.assertNotIn('verification', base['model'])
        for name in ('flash17-eaq-v1-exl3-2.9bpw', 'flash17-libertai-exl3-2.9bpw'):
            recipe = load_recipe(ROOT, name)
            self.assertEqual(recipe['engram'], base['engram'])
            self.assertEqual(recipe['serve'], base['serve'])
            self.assertEqual(recipe['env'], {**base['env'], 'TF_DSV41_NEEDS_MODEL': '/model'})
            generated = plan(ROOT, recipe, cluster)
            self.assertNotIn(generated['asset_key'], assets)
            assets.add(generated['asset_key'])
            for rank in (0, 1):
                env = generated['nodes'][rank]['env']
                self.assertEqual(env['TF_DSV41_NEEDS_MODEL'], model_dir(cluster['nodes'][rank], recipe['model'], True))
                self.assertNotEqual(generated['nodes'][rank]['paths']['prepared'], old['nodes'][rank]['paths']['prepared'])
        bad = copy.deepcopy(base)
        bad['model']['verification'] = 'skip'
        with self.assertRaises(StackError):
            validate_recipe(bad)
