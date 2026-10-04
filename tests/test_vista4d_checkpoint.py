"""CPU-only checkpoint conversion/loading tests. Never touch real checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from scripts.convert_vista4d_checkpoint import convert_checkpoint, parse_size, verify_conversion
from utils.vista4d_checkpoint import (
    checkpoint_sha256, checkpoint_size, load_checkpoint_into_model, load_pth,
    resolve_checkpoint, safetensors_shapes, wrapped_key_map,
)


class TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        self.latent_encoder = torch.nn.Linear(4, 4)
        self.base_only = torch.nn.Linear(4, 4)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.latent_encoder(x) + self.base_only(x)


class Wrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'dit.pth'
        self.output = self.root/'converted'
        torch.manual_seed(17)
        self.model = TinyDiT()
        self.state = {k: v.clone() for k, v in self.model.state_dict().items() if not k.startswith('base_only.')}
        torch.save(self.state, self.source)
        (self.root/'config.yaml').write_text('dit: {}\n')

    def convert(self, **kwargs):
        return convert_checkpoint(self.source, self.output, max_shard_size='80B', **kwargs)

    def index(self):
        path = self.output/'model.safetensors.index.json'
        return path, json.loads(path.read_text())

    def test_roundtrip_fp32_and_shard_payload_limit(self):
        before = self.source.read_bytes()
        report = self.convert()
        self.assertTrue(report['byte_exact_verified'])
        self.assertGreater(report['shards'], 1)
        self.assertEqual(self.source.read_bytes(), before)
        spec = resolve_checkpoint(self.output)
        self.assertEqual(set(spec.weight_map), set(self.state))
        for shard in spec.shards:
            state = load_file(str(shard))
            self.assertLessEqual(sum(t.numel()*t.element_size() for t in state.values()), 80)
            self.assertTrue(all(t.dtype == torch.float32 for t in state.values()))
        self.assertEqual((self.output/'config.yaml').read_bytes(), (self.root/'config.yaml').read_bytes())
        verify_conversion(self.state, self.output)

    def test_partial_overlay_forward_equivalence_and_streaming(self):
        self.convert()
        original, converted = TinyDiT(), TinyDiT()
        converted.load_state_dict(original.state_dict())
        base = converted.base_only.weight.clone()
        missing, unexpected = load_checkpoint_into_model(original, self.source)
        self.assertEqual(missing, ['base_only.bias', 'base_only.weight'])
        self.assertEqual(unexpected, [])
        with patch.object(converted, 'load_state_dict', wraps=converted.load_state_dict) as loader:
            self.assertEqual(load_checkpoint_into_model(converted, self.output), (missing, []))
            self.assertEqual(loader.call_count, len(resolve_checkpoint(self.output).shards))
        self.assertTrue(torch.equal(converted.base_only.weight, base))
        inputs = torch.randn(2, 4)
        self.assertTrue(torch.equal(original(inputs), converted(inputs)))

    def test_single_safetensors_and_directory(self):
        save_file(self.state, str(self.root/'model.safetensors'))
        load_checkpoint_into_model(self.model, self.root/'model.safetensors')
        with self.assertRaisesRegex(ValueError, 'Expected one'):
            resolve_checkpoint(self.root)
        single = self.root/'single'
        single.mkdir()
        save_file(self.state, str(single/'model.safetensors'))
        self.assertEqual(resolve_checkpoint(single).path, single/'model.safetensors')
        self.assertEqual(checkpoint_sha256(single), hashlib.sha256((single/'model.safetensors').read_bytes()).hexdigest())

    def test_wrapped_name_mapping(self):
        self.convert()
        model = TinyDiT()
        for i, block in enumerate(model.blocks):
            model.blocks[i] = Wrapper(block)
        model.blocks[0].module = Wrapper(model.blocks[0].module)
        mapping = wrapped_key_map(model)
        load_checkpoint_into_model(model, self.output, mapping)
        self.assertTrue(torch.equal(model.blocks[0].module.module.weight, self.state['blocks.0.weight']))

    def test_preserves_target_dtype(self):
        self.convert()
        model = TinyDiT().to(torch.bfloat16)
        load_checkpoint_into_model(model, self.output)
        self.assertEqual(model.blocks[0].weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(model.blocks[0].weight, self.state['blocks.0.weight'].to(torch.bfloat16)))

    def test_actual_offload_wrapper_on_cpu(self):
        from diffsynth.core import AutoWrappedModule
        self.convert()
        model = TinyDiT()
        for i, block in enumerate(model.blocks):
            model.blocks[i] = AutoWrappedModule(block, computation_device='cpu', computation_dtype=torch.float32)
        load_checkpoint_into_model(model, self.output, wrapped_key_map(model))
        self.assertTrue(torch.equal(model.blocks[0].module.weight, self.state['blocks.0.weight']))

    def test_late_shard_shape_failure_does_not_partially_mutate_model(self):
        self.convert()
        shard = resolve_checkpoint(self.output).shards[-1]
        state = load_file(str(shard))
        state[next(iter(state))] = torch.zeros(37)
        save_file(state, str(shard))
        with patch.object(self.model, 'load_state_dict') as load:
            with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
                load_checkpoint_into_model(self.model, self.output)
            load.assert_not_called()

    def test_legacy_hash_and_size_unchanged(self):
        expected = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.assertEqual(checkpoint_sha256(self.source), expected)
        self.assertEqual(checkpoint_sha256(self.root), expected)
        self.assertEqual(checkpoint_size(self.root), self.source.stat().st_size)

    def test_sharded_hash_covers_index_and_every_shard(self):
        self.convert()
        spec = resolve_checkpoint(self.output)
        fingerprint = checkpoint_sha256(spec.path)
        self.assertEqual(fingerprint, checkpoint_sha256(self.output))
        self.assertEqual(checkpoint_size(self.output), sum(p.stat().st_size for p in spec.files))
        (self.output/'unrelated.txt').write_text('not a weight')
        self.assertEqual(fingerprint, checkpoint_sha256(self.output))
        for shard in spec.shards:
            before = shard.read_bytes()
            data = load_file(str(shard))
            next(iter(data.values())).add_(1)
            save_file(data, str(shard))
            self.assertNotEqual(fingerprint, checkpoint_sha256(self.output))
            shard.write_bytes(before)
        spec.path.write_text(spec.path.read_text()+'\n')
        self.assertNotEqual(fingerprint, checkpoint_sha256(self.output))

    def test_missing_shard_rejected(self):
        self.convert()
        resolve_checkpoint(self.output).shards[0].unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unsafe'):
            resolve_checkpoint(self.output)

    def test_incorrect_index_and_extra_keys_rejected_before_copy(self):
        self.convert()
        path, index = self.index()
        key = next(iter(index['weight_map']))
        del index['weight_map'][key]
        path.write_text(json.dumps(index))
        with patch.object(self.model, 'load_state_dict') as load:
            with self.assertRaisesRegex(ValueError, 'keys mismatch'):
                load_checkpoint_into_model(self.model, self.output)
            load.assert_not_called()

    def test_duplicate_across_shards_rejected(self):
        self.convert()
        first, second, *_ = resolve_checkpoint(self.output).shards
        state = load_file(str(second))
        state.update(load_file(str(first)))
        save_file(state, str(second))
        with self.assertRaisesRegex(ValueError, 'keys mismatch'):
            safetensors_shapes(resolve_checkpoint(self.output))

    def test_duplicate_json_keys_rejected(self):
        self.convert()
        path, _ = self.index()
        path.write_text('{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            resolve_checkpoint(path)

    def test_traversal_and_symlink_escape_rejected(self):
        self.convert()
        path, index = self.index()
        for name in ('../escape.safetensors', '/tmp/escape.safetensors', 'sub\\escape.safetensors'):
            path.write_text(json.dumps({'weight_map': {'x': name}}))
            with self.assertRaisesRegex(ValueError, 'Unsafe shard'):
                resolve_checkpoint(path)
        save_file({'x': torch.zeros(1)}, str(self.root/'outside.safetensors'))
        (self.output/'escape.safetensors').symlink_to(self.root/'outside.safetensors')
        path.write_text(json.dumps({'weight_map': {'x': 'escape.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'unsafe shard'):
            resolve_checkpoint(path)

    def test_empty_or_bad_index_rejected(self):
        self.convert()
        path, _ = self.index()
        for mapping in ({}, [], None):
            path.write_text(json.dumps({'weight_map': mapping}))
            with self.assertRaises(ValueError):
                resolve_checkpoint(path)
        path.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'JSON object'):
            resolve_checkpoint(path)

    def test_numbered_shard_cannot_be_loaded_alone(self):
        self.convert()
        with self.assertRaisesRegex(ValueError, 'index/directory'):
            resolve_checkpoint(resolve_checkpoint(self.output).shards[0])

    def test_unexpected_shape_and_mapping_collision(self):
        self.convert()
        for bad_state, message in (({'wrong': torch.zeros(1)}, 'Unexpected'),
                                   ({'blocks.0.weight': torch.zeros(1)}, 'Shape mismatch')):
            path = self.root/'bad.safetensors'
            save_file(bad_state, str(path))
            with patch.object(self.model, 'load_state_dict') as load:
                with self.assertRaisesRegex(ValueError, message):
                    load_checkpoint_into_model(self.model, path)
                load.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'collision'):
            load_checkpoint_into_model(self.model, self.output,
                                       {'blocks.0.weight': 'blocks.1.weight'})

    def test_scalar_noncontiguous_alias_and_nan_preserved(self):
        tensor = torch.tensor([float('nan'), float('inf'), -0.0, 1.0])
        state = {'float': tensor, 'alias': tensor, 'scalar': torch.tensor(8),
                 'noncontiguous': torch.arange(12, dtype=torch.float32).reshape(3, 4).T,
                 'bool': torch.tensor([True]), 'empty': torch.empty(0)}
        torch.save(state, self.source)
        self.convert()
        verify_conversion(state, self.output)

    def test_verification_detects_changed_value(self):
        self.convert()
        spec = resolve_checkpoint(self.output)
        state = load_file(str(spec.shards[0]))
        next(iter(state.values())).add_(1)
        save_file(state, str(spec.shards[0]))
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            verify_conversion(self.state, self.output)

    def test_existing_output_and_oversized_tensor_refused(self):
        with self.assertRaisesRegex(ValueError, 'exceeds shard limit'):
            convert_checkpoint(self.source, self.output, '1B')
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.convert()

    def test_failed_verification_not_published(self):
        with patch('scripts.convert_vista4d_checkpoint.verify_conversion', side_effect=ValueError('bad')):
            with self.assertRaisesRegex(ValueError, 'bad'):
                self.convert()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.converted.tmp-*')), [])
        self.assertTrue(self.source.is_file())

    def test_old_serialization_pth_supported(self):
        torch.save(self.state, self.source, _use_new_zipfile_serialization=False)
        load_checkpoint_into_model(self.model, self.source)
        self.convert()

    def test_non_tensor_checkpoint_rejected(self):
        torch.save({'optimizer': {}}, self.source)
        with self.assertRaisesRegex(ValueError, 'flat tensor'):
            load_pth(self.source)

    def test_units(self):
        self.assertEqual(parse_size('4GB'), 4_000_000_000)
        self.assertEqual(parse_size('4GiB'), 4 * 2**30)
        for value in ('0', '-1', '4garbage'):
            with self.assertRaises(ValueError):
                parse_size(value)

    def test_cli_and_lazy_import(self):
        result = subprocess.run([sys.executable, '-m', 'scripts.convert_vista4d_checkpoint',
                                 '--input', str(self.source), '--output', str(self.output),
                                 '--max-shard-size', '80B'], check=True, capture_output=True, text=True)
        self.assertTrue(json.loads(result.stdout)['byte_exact_verified'])
        subprocess.run([sys.executable, '-c',
                        'import sys; from utils.vista4d_checkpoint import resolve_checkpoint; '
                        'resolve_checkpoint(sys.argv[1]); assert "torch" not in sys.modules',
                        str(self.output)], check=True)

    def test_flowlong_wrapper_completed_legacy_and_sharded_fingerprints(self):
        self.convert()
        outputs = self.root/'outputs'
        outputs.mkdir()
        (outputs/'video_seed=10027.mp4').write_bytes(b'completed-placeholder')
        manifest = self.root/'demo_splits_manifest.json'
        manifest.write_text('{}')  # Completed branch must not reach conditions/model loading.
        env = {**os.environ, 'PATH': str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],
               'VISTA4D_FOLDER': str(self.output), 'OUTPUT_FOLDER': str(outputs),
               'RESOLUTION': '384p', 'SEEDS': '10027', 'FORCE': 'false'}
        for checkpoint in (self.source, self.output):
            digest = checkpoint_sha256(checkpoint)
            env['VISTA4D_CHECKPOINT'] = str(checkpoint)
            env.pop('VISTA4D_CHECKPOINT_SHA256', None)
            report = outputs/'flowlong_report_seed=10027.json'
            report.write_text(json.dumps({'model': {'vista4d_checkpoint': {'sha256': digest}}}))
            result = subprocess.run(['bash', 'scripts/test_video/run_flowlong_inference.sh', str(manifest)],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('already complete', result.stdout)
            report.write_text(json.dumps({'model': {'vista4d_checkpoint': {'sha256': '0'*64}}}))
            result = subprocess.run(['bash', 'scripts/test_video/run_flowlong_inference.sh', str(manifest)],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('different/unknown checkpoint', result.stderr)


if __name__ == '__main__':
    unittest.main()
