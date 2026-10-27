"""Protect shared artifact identity and existing users' local data."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedResourcesTest(unittest.TestCase):
    def fixture(self, base):
        project, shared = base / 'project', base / 'shared'
        (project / 'config').mkdir(parents=True)
        hls = project / 'full-decode-stream/hls'
        hls.mkdir(parents=True)
        (hls / 'full_decode.cpp').write_bytes(b'kernel-source')
        (hls / 'full_decode.hpp').write_bytes(b'kernel-header')
        shared.mkdir()
        blob = shared / 'weights.bin'
        blob.write_bytes(b'original verified weight data')
        record = {'schema_version': 1, 'model_id': 'test-model', 'revision': 'fixed',
            'kernel_source_sha256': hashlib.sha256(b'kernel-sourcekernel-header').hexdigest(),
            'quantized': False,
            'checksums': {'weights': {'path': str(blob), 'sha256': hashlib.sha256(blob.read_bytes()).hexdigest()}}}
        for key in ['vitis_settings', 'platform', 'xrt_lib']:
            p = shared / key
            p.write_text('present')
            record[key] = str(p)
        include = shared / 'include/xrt'
        include.mkdir(parents=True)
        (include / 'xrt_device.h').write_text('header')
        record['xrt_include'] = str(include.parent)
        for key in ['packed_weights', 'reference']:
            p = shared / key
            p.mkdir()
            record[key] = str(p)
        (shared / 'resources.json').write_text(json.dumps(record))
        (project / 'config/shared-server.example.json').write_text(json.dumps(record))
        module = load_module('shared_resource_under_test', ROOT / 'scripts/use_shared_resources.py')
        module.ROOT = project
        return project, shared, blob, module

    def test_changed_shared_weight_is_rejected_before_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            project, shared, blob, module = self.fixture(Path(directory))
            blob.write_bytes(b'accidentally overwritten')
            with patch('sys.argv', ['use_shared_resources', '--shared-root', str(shared)]):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    module.main()
            self.assertFalse((project / 'deployment.json').exists())

    def test_existing_personal_weights_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            project, shared, _, module = self.fixture(Path(directory))
            personal = project / 'full-decode-stream/weights'
            personal.mkdir()
            (personal / 'keep.bin').write_bytes(b'personal data')
            with patch('sys.argv', ['use_shared_resources', '--shared-root', str(shared)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(FileExistsError, 'preserved'):
                        module.main()
            self.assertEqual((personal / 'keep.bin').read_bytes(), b'personal data')
            self.assertFalse((project / 'deployment.json').exists())

    def test_changed_shared_bitstream_is_rejected_before_hardware_access(self):
        module = load_module('shared_inference_under_test', ROOT / 'full-decode-stream/scripts/run_full.py')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            module.PROJECT = base / 'full-decode-stream'
            blob = base / 'shared.xclbin'
            blob.write_bytes(b'unexpected bitstream')
            config = {'shared_resources': {'xclbin': str(blob),
                'xclbin_sha256': hashlib.sha256(b'verified bitstream').hexdigest()}}
            (base / 'deployment.json').write_text(json.dumps(config))
            stderr = io.StringIO()
            with patch('sys.argv', ['run_full', '--backend', 'fpga']):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        module.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn('Shared bitstream SHA256 differs', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
