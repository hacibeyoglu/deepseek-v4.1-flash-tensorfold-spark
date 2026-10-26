"""Offline transfer tests: small real cache copies plus mocked SSH/Docker boundaries."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from lib import cache_sync, node
from lib.cluster import Cluster
from lib.config import StackError, load_recipe, synthetic_cluster

ROOT = Path(__file__).resolve().parents[1]


class CacheTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.head, self.worker = root / 'head cache', root / 'worker cache'
        self.recipe = {'model': {'repo': 'org/model', 'revision': 'a' * 40},
                       'engram': {'repo': 'org/original', 'revision': 'b' * 40}}
        self.receipt = {**self.recipe, 'engram_files': ['embedding.bin'],
                        'engram_sha256': {'embedding.bin': hashlib.sha256(b'engram').hexdigest()}}
        for key in ('model', 'engram'):
            folder = node.snapshot(self.head, self.recipe[key])
            folder.mkdir(parents=True)
            (folder / 'config.json').write_text('{}')
            (folder / 'model.safetensors.index.json').write_text('{}')
            blob = folder.parents[1] / 'blobs' / 'blob'
            blob.parent.mkdir()
            blob.write_bytes(b'model' if key == 'model' else b'engram')
            name = 'weights.bin' if key == 'model' else 'embedding.bin'
            (folder / name).symlink_to('../../blobs/blob')
            if key == 'model':
                (folder / 'SHA256SUMS').write_text(hashlib.sha256(b'model').hexdigest() + '  weights.bin\n')
        # A shared hub-level blob, as seen in older caches; preserve both links.
        shared = self.head / 'blobs/shared'
        shared.parent.mkdir()
        shared.write_bytes(b'shared')
        (self.head / 'models--org--model/blobs/shared').symlink_to('../../blobs/shared')
        (node.snapshot(self.head, self.recipe['model']) / 'tokenizer.json').symlink_to('../../blobs/shared')
        (self.head / 'token').write_text('must-not-copy')
        (self.head / 'models--other').mkdir()
        (self.head / 'models--other/file').write_text('unrelated')
        self.entries = cache_sync.manifest(self.head, self.recipe, self.receipt)

    def copy(self):
        listing = ''.join(e['path'] + '\0' for e in self.entries).encode()
        subprocess.run(['rsync', '-a', '--checksum', '--from0', '--files-from=-',
                        str(self.head) + '/', str(self.worker) + '/'], input=listing, check=True)

    def test_real_rsync_preserves_links_excludes_secrets_and_repairs_corruption(self):
        cache_sync.prepare(self.worker, self.entries)
        (self.worker / 'keep').write_text('unrelated')
        self.copy()
        cache_sync.verify(self.worker, self.entries)
        self.assertFalse((self.worker / 'token').exists())
        self.assertFalse((self.worker / 'models--other').exists())
        self.assertEqual((self.worker / 'keep').read_text(), 'unrelated')
        link = node.snapshot(self.worker, self.recipe['model']) / 'weights.bin'
        self.assertTrue(link.is_symlink())
        link.write_bytes(b'wrong')  # Same length corruption must not be skipped.
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            cache_sync.verify(self.worker, self.entries)
        self.copy()
        cache_sync.verify(self.worker, self.entries)
        self.copy()  # Repeat transfers remain valid.
        cache_sync.verify(self.worker, self.entries)

    def test_missing_blob_and_unsafe_destination_rejected(self):
        self.worker.mkdir()
        (self.worker / 'models--org--model').symlink_to(self.head / 'models--org--model')
        with self.assertRaisesRegex(RuntimeError, 'Symlink'):
            cache_sync.prepare(self.worker, self.entries)
        with self.assertRaises(ValueError):
            cache_sync.prepare(self.worker, [{'path': '../escape', 'size': 1}])

    def test_insufficient_space_fails_before_copy(self):
        with patch('lib.cache_sync.shutil.disk_usage', return_value=Mock(free=0)):
            with self.assertRaisesRegex(RuntimeError, 'lacks space'):
                cache_sync.prepare(self.worker, self.entries)

    def test_offline_receipt_only_written_after_verification(self):
        self.worker.mkdir()
        request = {'action': 'cache-verify', 'root': str(self.worker), 'cache': str(self.worker),
                   'entries': self.entries, 'receipt': self.receipt}
        with self.assertRaises(RuntimeError):
            node.main(request)
        self.assertFalse((self.worker / '.stack/download.json').exists())
        self.copy()
        with patch('lib.hub.authenticated_client', side_effect=AssertionError('worker accessed Hub')):
            node.main(request)
        self.assertEqual(json.loads((self.worker / '.stack/download.json').read_text())['source'], 'head-fabric-sync')


class OrchestrationTransferTests(unittest.TestCase):
    def setUp(self):
        config = synthetic_cluster(ROOT, '192.0.2.11,192.0.2.12')
        config['nodes'][1]['identity'] = 'worker-id'
        config['nodes'][1]['target'] = 'management-alias'
        self.c = Cluster(ROOT, load_recipe(ROOT, 'flash17-exl3-2.8bpw'), config)
        self.c.remotes = [Mock(), Mock()]
        self.c.doctor = Mock(return_value={'containers': [{'state': 'absent'}] * 2,
                                          'probes': [{}, {'username': 'user'}]})
        self.target = 'user@192.0.2.12'

    def test_fabric_target_verified_instead_of_management_alias(self):
        with patch('lib.cluster.Remote') as remote:
            remote.return_value.call.return_value = {'identity': 'worker-id'}
            self.assertEqual(self.c.transfer_target(), self.target)
            self.assertEqual(remote.call_args.args[1]['target'], self.target)
        self.c.remotes[0].call.assert_called_once_with('route', routes=[('192.0.2.12', 'rdma0')])

    def test_wrong_identity_or_unreachable_fabric_has_no_fallback(self):
        with patch('lib.cluster.Remote') as remote:
            remote.return_value.call.return_value = {'identity': 'wrong'}
            with self.assertRaisesRegex(StackError, 'different worker'):
                self.c.transfer_target()
            remote.return_value.call.side_effect = StackError('unreachable')
            with self.assertRaisesRegex(StackError, 'unreachable'):
                self.c.transfer_target()

    def test_only_head_downloads_and_failed_copy_never_verifies_worker(self):
        self.c.transfer_target = Mock(return_value=self.target)
        self.c.sync_code = Mock()
        self.c.remotes[0].call.return_value = {'receipt': True}
        entries = [{'path': 'models--org--model/blobs/file'}]
        with patch('lib.cluster.cache_sync.manifest', return_value=entries), patch('lib.cluster.sources.run') as run:
            self.c.download()
            command = run.call_args.args[0]
            self.assertTrue(command[-1].startswith(self.target + ':'))
            self.assertIn('--checksum', command)
            self.assertNotIn('--delete', command)
            self.assertEqual([c.args[0] for c in self.c.remotes[1].call.call_args_list], ['cache-prepare', 'cache-verify'])
            self.assertEqual(self.c.remotes[0].call.call_args.args[0], 'download')
            self.c.remotes[1].reset_mock()
            run.side_effect = RuntimeError('copy failed')
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                self.c.download()
            self.assertEqual([c.args[0] for c in self.c.remotes[1].call.call_args_list], ['cache-prepare'])

    def test_failed_fabric_check_prevents_download(self):
        self.c.transfer_target = Mock(side_effect=StackError('fabric failed'))
        with self.assertRaises(StackError):
            self.c.download()
        self.c.remotes[0].call.assert_not_called()

    def test_image_skips_identical_and_streams_missing_over_fabric(self):
        self.c.remotes[0].call.return_value = {'id': 'sha256:expected'}
        self.c.remotes[1].call.return_value = {'id': 'sha256:expected'}
        with patch('lib.cluster.subprocess.Popen') as save:
            self.c.sync_image('runtime:tag', self.target)
            save.assert_not_called()
        self.c.remotes[1].call.side_effect = [None, {'id': 'sha256:expected'}]
        with patch('lib.cluster.subprocess.Popen') as save, patch('lib.cluster.subprocess.run') as load:
            save.return_value.wait.return_value = 0
            load.return_value.returncode = 0
            self.c.sync_image('registry/asset@sha256:digest', self.target)
            self.assertEqual(save.call_args.args[0], ['docker', 'save', 'sha256:expected'])
            self.assertEqual(load.call_args.args[0][-2:], [self.target, 'docker load'])
            self.assertEqual(self.c.remotes[1].call.call_args.kwargs['image'], 'sha256:expected')

    def test_image_transfer_failure_and_mismatch_fail(self):
        self.c.remotes[0].call.return_value = {'id': 'sha256:expected'}
        for save_status, load_status, final_id in ((1, 0, 'expected'), (0, 1, 'expected'), (0, 0, 'wrong')):
            self.c.remotes[1].call.side_effect = [None, {'id': 'sha256:' + final_id}]
            with patch('lib.cluster.subprocess.Popen') as save, patch('lib.cluster.subprocess.run') as load:
                save.return_value.wait.return_value = save_status
                load.return_value.returncode = load_status
                with self.assertRaises(StackError):
                    self.c.sync_image('runtime:tag', self.target)

    def test_prepare_pulls_asset_only_on_head_and_uses_copied_id(self):
        self.c.states = Mock(return_value=[{'state': 'absent'}] * 2)
        self.c.transfer_target = Mock(return_value=self.target)
        self.c.sync_code = Mock()
        self.c.copy_dir = Mock()
        self.c.pair = Mock(return_value=[{'id': 'same'}] * 2)
        self.c.sync_image = Mock(return_value='sha256:asset')
        with patch('lib.cluster.sources.verify'):
            self.c.prepare()
        head_calls = self.c.remotes[0].call.call_args_list
        self.assertEqual(head_calls[0].kwargs['argv'][:2], ['docker', 'pull'])
        self.assertEqual(head_calls[1].kwargs['asset_image'], 'sha256:asset')
        self.assertEqual([c.args[0] for c in self.c.remotes[1].call.call_args_list], ['prepare'])
        self.assertEqual(self.c.remotes[1].call.call_args.kwargs['asset_image'], 'sha256:asset')
