import tempfile
from pathlib import Path
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from diffsynth.core import AutoWrappedModule
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.models.latent_encoder import LatentEncoder
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video_vista4d import (
    Vista4DPipeline, WanVideoUnit_CfgMerger, _slice_flowlong_batch, model_fn_vista4d,
)
from jepa.adapter import AdapterConfig, JEPAAdapter
from jepa.training import adapter_flow_loss, prepare_adapter_example


def tiny_dit():
    dit = WanModel(dim=32, in_dim=2, out_dim=2, ffn_dim=64, freq_dim=16, text_dim=8,
                   eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2, has_image_input=False)
    dit.positional_embedding_offset = 4
    dit.latent_encoder = LatentEncoder(wan_patch_embedding=dit.patch_embedding,
                                      use_source_masks=False, use_point_cloud_masks=False, rgb_in_channels=2)
    return dit.eval().requires_grad_(False)


class JEPAPipelineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dit = tiny_dit()
        self.inputs = {"source_video_latents": torch.randn(2, 2, 2, 4, 4),
                       "point_cloud_video_latents": torch.randn(2, 2, 2, 4, 4),
                       "context": torch.randn(2, 3, 8)}
        self.latents = torch.randn(2, 2, 2, 4, 4)
        self.features = torch.randn(2, 8, 2, 3, 3)

    def attach(self):
        self.dit.jepa_adapter = JEPAAdapter(32, 2, AdapterConfig(8, 16, 2, (0, 1), 2))

    def forward(self, **extra):
        return model_fn_vista4d(self.dit, **self.inputs, latents=self.latents, timestep=torch.tensor([500., 500.]), **extra)

    def test_zero_init_is_exact_baseline_and_adapter_only_receives_gradient(self):
        baseline = self.forward()
        self.attach()
        output = self.forward(jepa_features=self.features, use_gradient_checkpointing=True)
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        loss = output.square().mean()
        loss.backward()
        adapter = self.dit.jepa_adapter
        self.assertGreater(adapter.branches["0"].output.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for n, p in self.dit.named_parameters() if not n.startswith("jepa_adapter.")))
        optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
        optimizer.step()
        self.assertFalse(torch.allclose(self.forward(jepa_features=self.features), baseline))
        torch.testing.assert_close(self.forward(jepa_features=self.features, jepa_scale=0), baseline, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.pt"
            adapter.save(path, "test")
            restored, encoder = JEPAAdapter.load(path, 32, 2)
            self.assertEqual(encoder, "test")
            expected = self.forward(jepa_features=self.features)
            self.dit.jepa_adapter = restored
            torch.testing.assert_close(self.forward(jepa_features=self.features), expected)

    def test_condition_changes_output_and_preserves_source_tokens(self):
        self.attach()
        torch.nn.init.normal_(self.dit.jepa_adapter.branches["0"].output.weight)
        x = torch.randn(2, 24, 32)
        features = self.dit.jepa_adapter.prepare(self.features, (2, 2, 2), 2)
        result = self.dit.jepa_adapter(x, features, (2, 2, 2), 0)
        torch.testing.assert_close(result[:, 8:], x[:, 8:], rtol=0, atol=0)
        self.assertFalse(torch.allclose(self.forward(jepa_features=self.features), self.forward(jepa_features=-self.features)))
        with self.assertRaisesRegex(ValueError, "aligned"):
            self.forward(jepa_features=self.features[:, :, :1])

    def test_cfg_merge_matches_separate_batches_with_multiple_seeds(self):
        self.attach()
        torch.nn.init.normal_(self.dit.jepa_adapter.branches["0"].output.weight, std=.1)
        positive, negative = {"context": self.inputs["context"]}, {"context": -self.inputs["context"]}
        shared = {**self.inputs, "jepa_features": self.features, "latents": self.latents, "cfg_merge": True}
        del shared["context"]
        merged, _, _ = WanVideoUnit_CfgMerger().process(None, shared, positive, negative)
        actual = model_fn_vista4d(self.dit, **merged, timestep=torch.tensor([500., 500.]))
        expected_pos = self.forward(jepa_features=self.features)
        self.inputs["context"] = -self.inputs["context"]
        expected_neg = self.forward(jepa_features=self.features)
        torch.testing.assert_close(actual, torch.cat((expected_pos, expected_neg)))

    def test_flowlong_microbatch_includes_jepa_features(self):
        self.attach()
        torch.nn.init.normal_(self.dit.jepa_adapter.branches["0"].output.weight, std=.1)
        expected = self.forward(jepa_features=self.features)
        values = {**self.inputs, "latents": self.latents, "jepa_features": self.features}
        outputs = [model_fn_vista4d(self.dit, **_slice_flowlong_batch(values, i, i + 1, 2), timestep=torch.tensor([500.])) for i in range(2)]
        torch.testing.assert_close(torch.cat(outputs), expected)

    def test_frozen_wrapped_block_and_flow_training_loss(self):
        self.attach()
        # The adapter lives outside the offload wrapper, so its weights/gradients
        # aren't lost when a wrapped block temporarily copies its computation module.
        self.dit.blocks[0] = AutoWrappedModule(self.dit.blocks[0], computation_dtype=torch.float32, computation_device="cpu")
        pipe = Vista4DPipeline(device="cpu", torch_dtype=torch.float32)
        pipe.dit, pipe.model_fn = self.dit, model_fn_vista4d
        pipe.scheduler.set_timesteps(10, training=True, shift=5)
        loss = adapter_flow_loss(pipe, self.inputs, self.latents, self.features)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(self.dit.jepa_adapter.branches["0"].output.weight.grad.abs().sum().item(), 0)

    def test_fp32_adapter_can_train_through_bfloat16_generator(self):
        self.dit.to(dtype=torch.bfloat16)
        self.attach()
        pipe = Vista4DPipeline(device="cpu", torch_dtype=torch.bfloat16)
        pipe.dit, pipe.model_fn = self.dit, model_fn_vista4d
        pipe.scheduler.set_timesteps(10, training=True, shift=5)
        inputs = {k: v.bfloat16() for k, v in self.inputs.items()}
        loss = adapter_flow_loss(pipe, inputs, self.latents.bfloat16(), self.features)
        loss.backward()
        grad = self.dit.jepa_adapter.branches["0"].output.weight.grad
        self.assertEqual(grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_training_target_is_encoded_separately_from_conditions(self):
        import numpy as np
        from PIL import Image
        from types import MethodType
        pipe = Vista4DPipeline(device="cpu", torch_dtype=torch.float32)
        conditioning_calls = []
        video = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        source = {"source_video": [[Image.fromarray(frame) for frame in video]],
                  "point_cloud_video": [[Image.fromarray(frame) for frame in video]]}
        def prepare(self, **kwargs):
            conditioning_calls.append(kwargs)
            return self_inputs, {"context": torch.ones(1, 2, 8)}, {}
        self_inputs = {"source_video_latents": torch.zeros(1, 2, 2, 2, 2)}
        pipe._prepare_inference_inputs = MethodType(prepare, pipe)
        pipe.load_models_to_device = lambda names: None
        class VAE(torch.nn.Module):
            def encode(self, x, **kwargs):
                return x
        pipe.vae = VAE()
        args = Namespace(seed=[1], num_frames=5, height=16, width=16, sigma_shift=5., tile_vae=False)
        with patch("scripts.inference.inference.get_inputs", return_value=(source, 24.)), \
             patch("utils.media.load_video", return_value=(video + 255, 24.)):
            inputs, target = prepare_adapter_example(pipe, args, {}, "target.mp4")
        self.assertEqual(len(conditioning_calls), 1)
        self.assertNotIn("target_video", conditioning_calls[0])
        self.assertNotIn("input_video", conditioning_calls[0])
        self.assertEqual(set(inputs), {"source_video_latents", "context"})
        torch.testing.assert_close(target, torch.ones(1, 3, 5, 16, 16))


if __name__ == "__main__":
    unittest.main()
