"""Real filesystem checks for zero-copy prepared-cache adoption; no cluster access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lib.config import load_recipe, plan, synthetic_cluster
from lib.node import reuse_prepared

ROOT = Path(__file__).resolve().parents[1]


class PreparedReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.recipe = load_recipe(ROOT, 'flash17-jay-exl3-2.8bpw')
        self.revision = self.recipe['model']['revision']
        self.cluster = synthetic_cluster(self.root, '192.0.2.11,192.0.2.12')
        self.node = plan(ROOT, self.recipe, self.cluster)['nodes'][0]
        self.base = self.root / '.stack/prepared'
        self.old = self.base / ('a' * 16) / self.revision / ('b' * 16) / 'rank0'
        self.old.mkdir(parents=True)
        self.target = Path(self.node['paths']['prepared']) / ('b' * 16) / 'rank0'

    def test_runtime_changes_reuse_root_but_keep_native_and_model_keys(self):
        first = plan(ROOT, self.recipe, self.cluster)
        with patch('lib.config.runtime_id', return_value='c' * 16):
            updated = plan(ROOT, self.recipe, self.cluster)
        self.assertNotEqual(first['asset_key'], updated['asset_key'])
        self.assertEqual(first['nodes'][0]['paths']['prepared'], updated['nodes'][0]['paths']['prepared'])
        self.recipe['model']['revision'] = 'd' * 40
        other = plan(ROOT, self.recipe, self.cluster)
        self.assertNotEqual(first['nodes'][0]['paths']['prepared'], other['nodes'][0]['paths']['prepared'])

    def test_complete_files_link_once_without_payload_reads_or_overwrite(self):
        for name in ('L0.bin', 'L42.bin', 'top.bin', 'L3.bin.tmp', 'notes'):
            (self.old / name).write_bytes(b'original')
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('no weight scans')):
            self.assertEqual(reuse_prepared(self.node, self.revision), 3)
            self.assertEqual(reuse_prepared(self.node, self.revision), 0)
        self.assertEqual(sorted(p.name for p in self.target.iterdir()), ['L0.bin', 'L42.bin', 'top.bin'])
        self.assertEqual((self.target / 'L0.bin').stat().st_ino, (self.old / 'L0.bin').stat().st_ino)
        # Native rewrites are atomic replacements, so legacy files remain intact.
        replacement = self.target / 'new.tmp'
        replacement.write_bytes(b'replacement')
        replacement.replace(self.target / 'L0.bin')
        self.assertEqual(reuse_prepared(self.node, self.revision), 0)
        self.assertEqual((self.target / 'L0.bin').read_bytes(), b'replacement')
        self.assertEqual((self.old / 'L0.bin').read_bytes(), b'original')

    def test_other_rank_revision_keys_and_symlinks_are_not_mixed(self):
        (self.old / 'L0.bin').write_bytes(b'rank0')
        other_rank = self.old.parent / 'rank1'
        other_rank.mkdir()
        (other_rank / 'top.bin').write_bytes(b'rank1')
        (self.old / 'L1.bin').symlink_to('L0.bin')
        (self.old.parent / ('c' * 16)).symlink_to(self.old, target_is_directory=True)
        wrong_revision = self.base / ('a' * 16) / ('d' * 40) / ('b' * 16) / 'rank0'
        wrong_revision.mkdir(parents=True)
        (wrong_revision / 'L2.bin').write_bytes(b'other model')
        self.assertEqual(reuse_prepared(self.node, self.revision), 1)
        self.assertEqual([p.name for p in self.target.iterdir()], ['L0.bin'])

    def test_missing_cache_and_legacy_plans_are_safe(self):
        self.assertEqual(reuse_prepared(self.node, self.revision), 0)
        legacy = {**self.node, 'paths': {**self.node['paths'], 'prepared': str(self.old.parent.parent)}}
        (self.old / 'L0.bin').write_bytes(b'old')
        self.assertEqual(reuse_prepared(legacy, self.revision), 0)


if __name__ == '__main__':
    unittest.main()
