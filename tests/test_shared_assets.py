"""Synthetic assets only: worker reuse must fail closed without planning or compilation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from lib import node
from lib.config import load_recipe, plan, synthetic_cluster

ROOT = Path(__file__).resolve().parents[1]


class SharedAssetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        recipe = load_recipe(ROOT, 'flash17-jay-exl3-2.8bpw')
        cfg = synthetic_cluster(self.root, '192.0.2.11,192.0.2.12')
        cfg['nodes'][1].update(root=str(self.root), hub_cache=str(self.root / 'hub'))
        generated = plan(ROOT, recipe, cfg)
        worker = generated['nodes'][1]
        self.request = dict(node=worker, recipe=recipe, runtime=generated['runtime'],
                            asset_key=generated['asset_key'], image=generated['image'], asset_image='tools')
        node.make_paths(worker)
        self.assets = Path(worker['paths']['assets'])
        for name in node.SHARED_ASSETS:
            p = self.assets / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'fixture')
        self.cubin = 'a' * 64
        (self.assets / 'aot/aot.json').write_text(json.dumps({'kernels': [{'hash': self.cubin}]}))
        (self.assets / 'aot/cubins').mkdir()
        (self.assets / f'aot/cubins/{self.cubin}.cubin').write_bytes(b'kernel')
        self.shared = dict(identity={k: self.request[k] for k in ('runtime', 'asset_key')},
                           hashes=node.shared_hashes(self.assets))
        model = node.snapshot(worker['hub_cache'], recipe['model'])
        model.mkdir(parents=True)
        (model / 'config.json').write_text(json.dumps({'engram_layer_ids': [1, 14]}))
        for layer in (1, 14):
            (Path(worker['paths']['engram']) / f'engram-l{layer}-r1of2.bin').write_bytes(b'packed')
        dense = self.root / '.stack/source/config/pfdense-table.json'
        dense.parent.mkdir(parents=True)
        dense.write_text('{}')

    def test_worker_reuses_all_assets_and_keeps_rank_local_engram_checks(self):
        with patch('lib.node.run', return_value=Mock(returncode=0)) as run:
            result = node.prepare({**self.request, 'shared': self.shared})
        commands = [call.args[0] for call in run.call_args_list]
        self.assertFalse(any('aot-needs' in c or 'compile' in c or 'sigs' in c for c in commands))
        pack = next(c for c in commands if '--check' in c)
        self.assertEqual(pack[pack.index('--rank') + 1], '1')
        marker = json.loads((self.assets / 'prepared.json').read_text())
        self.assertEqual(marker['identity']['rank'], 1)
        self.assertEqual(marker['hashes'], self.shared['hashes'])
        self.assertEqual(result['shared'], self.shared)

    def test_corrupt_cubin_or_table_never_produces_worker_receipt(self):
        for name in ('rope.bin', f'aot/cubins/{self.cubin}.cubin'):
            path = self.assets / name
            previous = path.read_bytes()
            path.write_bytes(b'corrupt')
            with patch('lib.node.run') as run, self.assertRaisesRegex(RuntimeError, 'differ'):
                node.prepare({**self.request, 'shared': self.shared})
            run.assert_not_called()
            self.assertFalse((self.assets / 'prepared.json').exists())
            path.write_bytes(previous)

    def test_wrong_identity_or_missing_receipt_rejected(self):
        bad = {**self.shared, 'identity': {'runtime': 'wrong', 'asset_key': self.request['asset_key']}}
        for extra in ({}, {'shared': bad}):
            with patch('lib.node.run') as run, self.assertRaises(RuntimeError):
                node.prepare({**self.request, **extra})
            run.assert_not_called()

    def test_missing_cubin_rejected(self):
        (self.assets / f'aot/cubins/{self.cubin}.cubin').unlink()
        with self.assertRaises(FileNotFoundError):
            node.verify_shared(self.assets, self.shared, self.request)

    def test_cached_head_corruption_is_not_recertified_or_shared(self):
        head = {**self.request['node'], 'rank': 0}
        request = {**self.request, 'node': head}
        for layer in (1, 14):
            (Path(head['paths']['engram']) / f'engram-l{layer}-r0of2.bin').write_bytes(b'packed')
        marker = {'identity': {**self.shared['identity'], 'rank': 0}, 'hashes': self.shared['hashes']}
        stamp = self.assets / 'prepared.json'
        stamp.write_text(json.dumps(marker))
        (self.assets / 'rope.bin').write_bytes(b'corrupt')
        with patch('lib.node.run', return_value=Mock(returncode=0)) as run:
            with self.assertRaisesRegex(RuntimeError, 'Cached head asset changed'):
                node.prepare(request)
        self.assertFalse(any('aot-needs' in c.args[0] for c in run.call_args_list))
        self.assertEqual(json.loads(stamp.read_text()), marker)
