"""Real local processes and pipes; no SSH, Docker, downloads, or GPU use."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from lib import processes
from lib.cluster import Remote

ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError('Timed out waiting for test process')


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / 'work.py'
        self.work.write_text('''import os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
(root / 'download.incomplete').write_bytes(b'keep partial data')
(root / 'ready').write_text(str(os.getpid()) + ' ' + str(os.getppid()))
while True:
    with (root / 'heartbeat').open('a') as f: f.write('x')
    time.sleep(0.03)
''')
        self.children = []
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        for proc in self.children:
            if proc.poll() is None:
                processes.stop(proc, grace=0.1)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream:
                    stream.close()
        ready = self.root / 'ready'
        if ready.exists():
            for pid in map(int, ready.read_text().split()):
                if alive(pid):
                    try: os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError: pass

    def driver(self, body):
        code = f'''import sys
sys.path.insert(0, {str(ROOT)!r})
from lib import processes
from lib.cluster import Remote, Cluster
from lib.config import load_recipe, synthetic_cluster
processes.install_signals()
try:
{body}
except KeyboardInterrupt:
    sys.exit(130)
'''
        p = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
        self.children.append(p)
        return p

    def operation(self):
        return (f'Remote({str(ROOT)!r}, {{"root": {str(self.root)!r}}}).call("run", '
                f'argv={[sys.executable, str(self.work), str(self.root)]!r})')

    def wait_ready(self):
        wait_for(lambda: (self.root / 'ready').exists() and (self.root / 'ready').stat().st_size > 0)
        return tuple(map(int, (self.root / 'ready').read_text().split()))

    def assert_stopped_and_reusable(self, pids):
        for pid in pids:
            wait_for(lambda pid=pid: not alive(pid), seconds=8)
        lock = self.root / '.stack/node.lock'
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual((self.root / 'download.incomplete').read_bytes(), b'keep partial data')
        result = Remote(ROOT, {'root': str(self.root)}).call('run', argv=[sys.executable, '-c', 'print("retry works")'], capture=True)
        self.assertEqual(result['stdout'].strip(), 'retry works')

    def test_ctrl_c_repeated_stops_stubborn_child_and_releases_lock(self):
        p = self.driver('    ' + self.operation())
        pids = self.wait_ready()
        p.send_signal(signal.SIGINT)
        time.sleep(0.15)
        p.send_signal(signal.SIGINT)
        stdout, stderr = p.communicate(timeout=10)
        self.assertEqual(p.returncode, 130, stderr.decode())
        self.assertNotIn(b'Traceback', stderr)
        self.assertTrue(all(not alive(pid) for pid in pids))
        self.assert_stopped_and_reusable(pids)

    def test_parent_death_cancels_agent_using_pipe_eof(self):
        p = self.driver('    ' + self.operation())
        pids = self.wait_ready()
        p.kill()  # No Python cleanup: exercise the same EOF path as a lost SSH connection.
        p.communicate(timeout=10)
        self.assert_stopped_and_reusable(pids)

    def test_download_ctrl_c_keeps_partial_files_without_success_receipt(self):
        # Exercise node.download itself with a fake Hub interpreter, never installing or fetching.
        interpreter = self.root / '.stack/venv/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text(f'''#!{sys.executable}
import pathlib, sys
if sys.argv[1] == '-c':
    print('fixture')
else:
    sys.argv = [{str(self.work)!r}, {str(self.root)!r}]
    exec(compile(pathlib.Path({str(self.work)!r}).read_text(), 'fake-hub', 'exec'))
''')
        interpreter.chmod(0o755)
        args = {'node': {'root': str(self.root), 'hf_home': str(self.root / 'hf'),
                         'hub_cache': str(self.root / 'hf/hub')},
                'recipe': {'model': {}, 'engram': {}}, 'hf_version': 'fixture'}
        p = self.driver(f'    Remote({str(ROOT)!r}, {{"root": {str(self.root)!r}}}).call("download", **{args!r})')
        pids = self.wait_ready()
        p.send_signal(signal.SIGINT)
        _, stderr = p.communicate(timeout=10)
        self.assertEqual(p.returncode, 130, stderr.decode())
        self.assertNotIn(b'Traceback', stderr)
        self.assertFalse((self.root / '.stack/download.json').exists())
        self.assert_stopped_and_reusable(pids)

    def test_worker_agent_cancels_on_connection_close_without_checkout(self):
        # Execute exactly the embedded bootstrap produced for SSH, locally without SSH.
        from unittest.mock import patch
        captured = {}
        def record(args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return subprocess.CompletedProcess(args, 0, '{"ok":true,"result":{}}')
        with patch('lib.cluster.processes.run', side_effect=record):
            Remote(ROOT, {'root': str(self.root), 'target': 'fake-worker'}).call(
                'run', argv=[sys.executable, str(self.work), str(self.root)])
        import shlex
        p = subprocess.Popen(shlex.split(captured['args'][-1]), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        self.children.append(p)
        p.stdin.write(captured['kwargs']['input'].encode())
        p.stdin.flush()
        pids = self.wait_ready()
        p.stdin.close()
        p.stdin = None
        _, stderr = p.communicate(timeout=10)
        self.assertEqual(p.returncode, 130, stderr.decode())
        self.assertNotIn(b'Traceback', stderr)
        self.assert_stopped_and_reusable(pids)

    def test_pair_ctrl_c_cancels_calls_in_threads(self):
        body = f'''    c = Cluster({str(ROOT)!r}, load_recipe({str(ROOT)!r}, "flash17-exl3-2.8bpw"),
                synthetic_cluster({str(ROOT)!r}, "192.0.2.1,192.0.2.2"))
    c.remotes = [Remote({str(ROOT)!r}, {{"root": r}}) for r in {[str(self.root), str(self.root / 'other')]!r}]
    c.pair("run", [{{"argv": {[sys.executable, str(self.work), str(self.root)]!r}}},
                   {{"argv": [sys.executable, "-c", "import time; time.sleep(60)"]}}])'''
        p = self.driver(body)
        pids = self.wait_ready()
        p.send_signal(signal.SIGINT)
        _, stderr = p.communicate(timeout=10)
        self.assertEqual(p.returncode, 130, stderr.decode())
        self.assert_stopped_and_reusable(pids)

    def test_pipeline_interrupt_stops_both_sides(self):
        producer = [sys.executable, str(self.work), str(self.root)]
        consumer = [sys.executable, '-c', 'import sys,time; sys.stdin.read(); time.sleep(60)']
        p = self.driver(f'    processes.pipeline({producer!r}, {consumer!r})')
        child, parent = self.wait_ready()
        p.send_signal(signal.SIGINT)
        _, stderr = p.communicate(timeout=10)
        self.assertEqual(p.returncode, 130, stderr.decode())
        wait_for(lambda: not alive(child))
        self.assertNotIn(b'Traceback', stderr)

    def test_pipeline_transfers_and_propagates_failures(self):
        target = self.root / 'copied'
        processes.pipeline([sys.executable, '-c', 'print("image bytes")'],
                           [sys.executable, '-c', f'import pathlib,sys; pathlib.Path({str(target)!r}).write_text(sys.stdin.read())'])
        self.assertEqual(target.read_text(), 'image bytes\n')
        with self.assertRaises(subprocess.CalledProcessError):
            processes.pipeline([sys.executable, '-c', 'import time; time.sleep(60)'],
                               [sys.executable, '-c', 'raise SystemExit(2)'])
        with self.assertRaises(subprocess.CalledProcessError):
            processes.pipeline([sys.executable, '-c', 'raise SystemExit(3)'],
                               [sys.executable, '-c', 'import sys; sys.stdin.read()'])

    def test_runner_drains_large_output_and_preserves_exit_status(self):
        result = processes.run([sys.executable, '-c', 'import sys,time; print("x"*100000); time.sleep(.3); sys.exit(7)'],
                               text=True, capture_output=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(result.stdout), 100001)
