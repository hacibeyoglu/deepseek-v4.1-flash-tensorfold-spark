"""Temporary test builders must be cleaned without touching caller-owned images."""
import contextlib
import io
import subprocess
import unittest
from unittest.mock import patch

from lib.config import StackError
from scripts import stack


class RuntimeBuilderTests(unittest.TestCase):
    def setUp(self):
        for name, kwargs in (
                ('build_context', {'return_value': '/ctx'}),
                ('run', {})):
            patcher = patch.object(stack.sources, name, **kwargs)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        patcher = patch.object(stack.processes, 'run', return_value=subprocess.CompletedProcess([], 0, '', ''))
        self.cleanup_image = patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_removes_only_the_unique_temporary_image(self):
        with stack.runtime_builder() as image:
            self.assertTrue(image.startswith('spark-ds41-tensorfold:test-builder-'))
            self.cleanup_image.assert_not_called()
        command = self.run.call_args.args[0]
        self.assertEqual(command[command.index('--target') + 1], 'build')
        self.assertEqual(command[command.index('-t') + 1], image)
        self.cleanup_image.assert_called_once_with(
            ['docker', 'image', 'rm', '--no-prune', image], capture_output=True, text=True)

    def test_test_failure_and_interrupt_still_remove_builder(self):
        for error in (RuntimeError('test failed'), KeyboardInterrupt()):
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                with stack.runtime_builder() as image:
                    raise error
            self.assertEqual(self.cleanup_image.call_args.args[0][-1], image)

    def test_failed_build_attempts_cleanup_without_masking_build_failure(self):
        self.run.side_effect = subprocess.CalledProcessError(2, ['docker', 'build'])
        self.cleanup_image.return_value = subprocess.CompletedProcess([], 1, '', 'Error: No such image: temporary')
        with self.assertRaises(subprocess.CalledProcessError):
            with stack.runtime_builder():
                self.fail('Failed build yielded an image')
        self.cleanup_image.assert_called_once()

    def test_explicit_image_is_neither_built_nor_removed(self):
        with stack.runtime_builder('existing:builder') as image:
            self.assertEqual(image, 'existing:builder')
        self.build_context.assert_not_called()
        self.run.assert_not_called()
        self.cleanup_image.assert_not_called()

    def test_cleanup_failure_is_reported_and_preserves_original_failure(self):
        self.cleanup_image.return_value = subprocess.CompletedProcess([], 1, '', 'Docker unavailable')
        with self.assertRaisesRegex(StackError, 'Could not remove temporary test image'):
            with stack.runtime_builder():
                pass
        with contextlib.redirect_stderr(io.StringIO()) as output, self.assertRaisesRegex(RuntimeError, 'test failed'):
            with stack.runtime_builder():
                raise RuntimeError('test failed')
        self.assertIn('Could not remove temporary test image', output.getvalue())
