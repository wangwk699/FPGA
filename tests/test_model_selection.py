"""Keep model geometry, memory capacity and bitstream selection consistent."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from model_profiles import canonical_model,expected_shapes,geometry,load_config,project_path,source_digest,verify_bitstream_binding,weight_bytes
from export_weights import expected_shapes as legacy_shapes


class ModelSelectionTest(unittest.TestCase):
    def test_zero_six_shapes_remain_identical_to_verified_legacy_export(self):
        self.assertEqual(expected_shapes('Qwen3-0.6B'),legacy_shapes(geometry('Qwen3-0.6B')))
        self.assertEqual(weight_bytes('Qwen3-0.6B'),1192099840)

    def test_one_seven_requires_eight_bank_weight_capacity(self):
        self.assertEqual(weight_bytes('Qwen3-1.7B'),3441149952)
        self.assertGreater(weight_bytes('Qwen3-1.7B'),4*512*1024**2)
        self.assertLess(weight_bytes('Qwen3-1.7B'),8*512*1024**2)
        shapes=expected_shapes('Qwen3-1.7B')
        self.assertEqual(shapes['model.layers.0.self_attn.q_proj.weight'],[2048,2048])
        self.assertEqual(shapes['model.layers.0.mlp.down_proj.weight'],[2048,6144])

    def test_model_names_and_isolated_workspaces(self):
        self.assertEqual(canonical_model('Qwen/Qwen3-1.7B'),'Qwen3-1.7B')
        self.assertEqual(canonical_model('0.6B'),'Qwen3-0.6B')
        self.assertNotEqual(project_path(ROOT,'Qwen3-0.6B'),project_path(ROOT,'Qwen3-1.7B'))
        with self.assertRaises(ValueError):canonical_model('Qwen3-4B')

    def test_missing_one_seven_config_never_falls_back_to_zero_six(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'deployment.json').write_text(json.dumps({'model_id':'Qwen/Qwen3-0.6B'}))
            with self.assertRaisesRegex(FileNotFoundError,'Qwen3-1.7B is not prepared'):
                load_config(p,'Qwen3-1.7B')

    def test_bitstream_for_another_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);blob=p/'wrong.xclbin';blob.write_bytes(b'one-seven-design')
            config={'verified_xclbin':{'sha256':hashlib.sha256(b'zero-six-design').hexdigest()}}
            with self.assertRaisesRegex(ValueError,'selected model'):
                verify_bitstream_binding(p,config,blob)

    def test_registered_bitstream_is_bound_to_current_kernel_source(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'hls').mkdir()
            (p/'hls/full_decode.cpp').write_text('verified source')
            (p/'hls/full_decode.hpp').write_text('verified geometry')
            blob=p/'model.xclbin';blob.write_bytes(b'verified model design')
            config={'kernel_source_sha256':source_digest(p),
                'verified_xclbin':{'sha256':hashlib.sha256(blob.read_bytes()).hexdigest()}}
            verify_bitstream_binding(p,config,blob)
            (p/'hls/full_decode.hpp').write_text('different model geometry')
            with self.assertRaisesRegex(ValueError,'selected model'):
                verify_bitstream_binding(p,config,blob)

    def test_wrong_weight_geometry_is_rejected_before_loading_native_code(self):
        spec=importlib.util.spec_from_file_location('model_runtime_under_test',ROOT/'full-decode-stream/scripts/full_runtime.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'manifest.json').write_text(json.dumps({'format':'qwen3-tiled-bf16-v2','quantized':False,
                'tensors':[{}]*310,'config':geometry('Qwen3-0.6B')}))
            with patch.object(module.ctypes,'CDLL',side_effect=AssertionError('Native code must not load')):
                with self.assertRaisesRegex(ValueError,'hidden_size=2048'):
                    module.FullRuntime('cpu-cpp',weights=p,model_name='Qwen3-1.7B')


if __name__=='__main__':unittest.main()
