"""Small CPU baseline tests; no 14B checkpoints or dataset downloads required."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from diffsynth.core import AutoWrappedModule, AutoWrappedLinear
from diffsynth.core.vram import enable_vram_management
from diffsynth.models.wan_video_dit import DiTBlock, Head, RMSNorm
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_manifold4d import Manifold4DModel, camera_tokens, two_stream_freqs
from diffsynth.pipelines.manifold4d import (
    balanced_flow_loss, flow_path, make_scheduler, manifold_prior, pool_video_mask, sample_manifold,
)
from diffsynth.pipelines.manifold4d_training import manifold_training_loss, validate_training_example
from diffsynth.pipelines.wan_video_manifold4d import Manifold4DPipeline, resize_camera_intrinsics


def tiny_model(dtype=torch.float32):
    base = WanModel(dim=32, in_dim=16, out_dim=16, ffn_dim=64, freq_dim=16, text_dim=8,
                    eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2, has_image_input=False).to(dtype)
    return Manifold4DModel(base, positional_embedding_offset=4)


def example():
    return {
        "target_latents": torch.randn(2, 16, 2, 4, 4),
        "render_latents": torch.randn(2, 16, 2, 4, 4),
        "source_latents": torch.randn(2, 16, 2, 4, 4),
        "render_mask": torch.rand(2, 2, 2, 4, 4),
        "source_mask": torch.rand(2, 2, 2, 4, 4),
        "target_motion_mask": torch.rand(2, 1, 2, 4, 4),
        "camera_embedding": torch.randn(2, 16, 6),
        "context": torch.randn(2, 3, 8),
        "empty_context": torch.randn(2, 3, 8),
    }


class PriorTest(unittest.TestCase):
    def test_coverage_limits_and_shared_fractional_noise(self):
        render = torch.tensor([2., 4., 6.]).reshape(1, 1, 1, 1, 3)
        noise = torch.tensor([3., -2., 5.]).reshape_as(render)
        alpha = torch.tensor([0., 1., .25]).reshape_as(render)
        prior = manifold_prior(render, alpha, noise)
        torch.testing.assert_close(prior, torch.tensor([3., 3.4, 5.625]).reshape_as(render))
        torch.testing.assert_close(manifold_prior(render, alpha, noise, sigma=1), alpha * render + noise)

    def test_prior_preserves_fp32_precision_and_rejects_invalid_masks(self):
        render = torch.zeros(1, 16, 1, 2, 2, dtype=torch.bfloat16)
        noise = torch.randn_like(render, dtype=torch.float32)
        alpha = torch.zeros(1, 1, 1, 2, 2)
        self.assertTrue(torch.equal(manifold_prior(render, alpha, noise), noise))
        for invalid in (alpha + 2, alpha + float("nan"), alpha.expand(1, 16, 1, 2, 2)):
            with self.assertRaises(ValueError):
                manifold_prior(render, invalid, noise)
        with self.assertRaises(ValueError):
            manifold_prior(render, alpha, noise, sigma=-1)

    def test_first_frame_mask_layout_and_fractional_coverage(self):
        mask = torch.zeros(1, 2, 9, 16, 16)
        mask[:, 0, 0] = 1
        mask[:, 0, 1] = 1
        mask[:, 0, 5:] = 1
        mask[:, 1, :, :8] = 1
        result = pool_video_mask(mask)
        self.assertEqual(result.shape, (1, 2, 3, 2, 2))
        torch.testing.assert_close(result[0, 0, :, 0, 0], torch.tensor([1., .25, 1.]))
        torch.testing.assert_close(result[0, 1, :, 0, 0], torch.ones(3))
        torch.testing.assert_close(result[0, 1, :, 1, 0], torch.zeros(3))
        self.assertEqual(pool_video_mask(mask[:, :, :1]).shape[2], 1)

    def test_pool_rejects_unaligned_timelines_and_pixels(self):
        for shape in ((1, 1, 8, 16, 16), (1, 1, 9, 17, 16)):
            with self.assertRaises(ValueError):
                pool_video_mask(torch.zeros(shape))

    def test_flow_velocity_sign_and_reverse_sampling_endpoint(self):
        target, prior = torch.randn(2, 16, 2, 4, 4), torch.randn(2, 16, 2, 4, 4)
        xt, velocity = flow_path(target, prior, torch.tensor([0., 1.]))
        torch.testing.assert_close(xt[0], target[0])
        torch.testing.assert_close(xt[1], prior[1])
        scheduler = make_scheduler(5, solver="euler")
        value = prior
        for step in scheduler.timesteps:
            value = scheduler.step(velocity, step, value)
        torch.testing.assert_close(value, target)
        with self.assertRaises(ValueError):
            flow_path(target, prior, -0.1)

    def test_region_balance_and_dynamic_weight(self):
        velocity = torch.zeros(1, 1, 1, 1, 4)
        prediction = torch.tensor([1., 1., 1., 3.]).reshape_as(velocity)
        coverage = torch.tensor([1., 1., 1., 0.]).reshape_as(velocity)
        self.assertEqual(balanced_flow_loss(prediction, velocity, coverage).item(), 5.)
        moving = 1 - coverage
        self.assertEqual(balanced_flow_loss(prediction, velocity, coverage, moving).item(), 9.5)
        self.assertEqual(balanced_flow_loss(torch.ones_like(velocity), velocity, torch.zeros_like(coverage)).item(), 1.)
        self.assertEqual(balanced_flow_loss(torch.ones_like(velocity), velocity, torch.ones_like(coverage), torch.ones_like(moving)).item(), 2.)

    def test_unipc_uses_official_wan_sigma_grid(self):
        scheduler = make_scheduler(20, shift=5)
        raw = np.linspace(float(np.float32(.999)), 0., 21)[:-1]
        expected = 5 * raw / (1 + 4 * raw)
        torch.testing.assert_close(scheduler.sigmas, torch.tensor(np.r_[expected, 0], dtype=torch.float32), rtol=0, atol=0)
        self.assertTrue(torch.equal(scheduler.timesteps, torch.tensor(expected * 1000).long()))
        value = torch.randn(1, 2, 1, 2, 2)
        for step in scheduler.timesteps:
            value = scheduler.step(torch.zeros_like(value), step, value, return_dict=False)[0]
        self.assertTrue(torch.isfinite(value).all())


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.model = tiny_model()
        self.data = example()

    def forward(self, model=None, **extra):
        data = self.data
        return (model or self.model)(data["target_latents"], torch.tensor([300., 700.]), data["context"],
            source_latents=data["source_latents"], render_mask=data["render_mask"],
            source_mask=data["source_mask"], camera_embedding=data["camera_embedding"], **extra)

    def test_two_stream_token_layout_and_time_policy(self):
        captured = []
        hook = self.model.base_model.blocks[0].register_forward_pre_hook(lambda _, args: captured.append(args))
        output = self.forward()
        hook.remove()
        x, _, modulation, freqs, cameras = captured[0]
        self.assertEqual(x.shape, (2, 16, 32))
        self.assertEqual(freqs.shape[0], 16)
        self.assertEqual(cameras.shape, (2, 16, 6))
        self.assertEqual(output.shape, self.data["target_latents"].shape)
        self.assertFalse(torch.allclose(modulation[0, :8], modulation[1, :8]))
        torch.testing.assert_close(modulation[0, 8:], modulation[1, 8:])
        # Source time defaults to 0 regardless of the current output time.
        _, honest = self.model._time_embeddings(torch.tensor([300., 700.]), torch.tensor([1000., 0.]), 8)
        self.assertFalse(torch.allclose(honest[:, 8:], modulation[:, 8:]))
        self.model.cond_stream_t = "shared"
        _, shared = self.model._time_embeddings(torch.tensor([300., 700.]), None, 8)
        torch.testing.assert_close(shared[:, :8], shared[:, 8:])

    def test_trainable_subset_and_checkpointed_backward(self):
        loss = manifold_training_loss(self.model, self.data, drop_probability=0,
                                     unconditional_probability=0, use_gradient_checkpointing=True)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertGreater(self.model.output_rgb_patch_embed.weight.grad.abs().sum().item(), 0)
        self.assertFalse(self.model.base_model.blocks[0].ffn[0].weight.requires_grad)
        self.assertFalse(self.model.base_model.blocks[0].cross_attn.q.weight.requires_grad)
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        before = self.model.output_rgb_patch_embed.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.output_rgb_patch_embed.weight))

    def test_render_dropout_restores_gaussian_endpoint_and_honest_source_time(self):
        seen = []
        class Recorder:
            def __call__(self, latent, time, context, **conditions):
                seen.append((latent, time, context, conditions))
                return torch.zeros_like(latent)
        eps = torch.randn_like(self.data["target_latents"])
        manifold_training_loss(Recorder(), self.data, time=1., noise=eps,
                               drop_probability=0., unconditional_probability=1.)
        xt, timestep, context, cond = seen[0]
        torch.testing.assert_close(xt, eps, rtol=0, atol=0)
        torch.testing.assert_close(timestep, torch.full((2,), 1000.))
        torch.testing.assert_close(context, self.data["empty_context"])
        for name in ("render_mask", "source_mask", "camera_embedding"):
            self.assertEqual(cond[name].count_nonzero().item(), 0)
        torch.testing.assert_close(cond["source_timestep"], torch.full((2,), 1000.))

    def test_zero_init_camera_and_mask_modules(self):
        for name in ("output_anchor_patch_embed", "source_mask_patch_embed"):
            self.assertEqual(getattr(self.model, name).weight.count_nonzero().item(), 0)
        for block in self.model.base_model.blocks:
            self.assertEqual(block.cam_encoder.weight.count_nonzero().item(), 0)
            torch.testing.assert_close(block.projector.weight, torch.eye(32))

    def test_official_checkpoint_roundtrip_and_missing_subset_rejection(self):
        with TemporaryDirectory() as folder:
            self.model.save_checkpoint(folder)
            model = deepcopy(self.model)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(1)
            model.load_checkpoint(folder)
            torch.testing.assert_close(self.forward(model), self.forward(), rtol=0, atol=0)
            file = Path(folder) / "conditioning_modules.pt"
            saved = torch.load(file, weights_only=True)
            del saved["output_rgb_patch_embed"]["weight"]
            torch.save(saved, file)
            with self.assertRaisesRegex(ValueError, "missing"):
                model.load_checkpoint(folder)

    def test_wrong_shapes_schema_and_vista_checkpoint_rejected(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                self.model.load_checkpoint(folder)
            self.model.save_checkpoint(folder)
            file = Path(folder) / "camera_encoder.pt"
            saved = torch.load(file, weights_only=True)
            saved["cam_encoders"]["0.proj.weight"] = torch.zeros(1, 6)
            torch.save(saved, file)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                self.model.load_checkpoint(folder)
            self.model.save_checkpoint(folder)
            file = Path(folder) / "conditioning_modules.pt"
            saved = torch.load(file, weights_only=True)
            saved["schema_version"] = 99
            torch.save(saved, file)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                self.model.load_checkpoint(folder)

    def test_wrapped_model_checkpoint_and_forward(self):
        expected = self.forward()
        wrapped = deepcopy(self.model)
        wrapped.base_model.blocks[0] = AutoWrappedModule(wrapped.base_model.blocks[0],
                                                        computation_device="cpu", computation_dtype=torch.float32)
        with TemporaryDirectory() as folder:
            self.model.save_checkpoint(folder)
            wrapped.load_checkpoint(folder)
            torch.testing.assert_close(self.forward(wrapped), expected, rtol=0, atol=0)
            wrapped.save_checkpoint(folder)
            self.model.load_checkpoint(folder)

    def test_bf16_forward_and_time_embeddings_stay_fp32(self):
        model = tiny_model(torch.bfloat16)
        emb, modulation = model._time_embeddings(torch.tensor([300., 700.]), None, 8)
        self.assertEqual(emb.dtype, torch.float32)
        self.assertEqual(modulation.dtype, torch.float32)
        output = self.forward(model)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())

    def test_dtype_conversion_after_construction(self):
        self.model.to(dtype=torch.bfloat16)
        self.assertEqual(self.model.compute_dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(self.forward()).all())

    def test_cpu_offload_wrappers_include_new_projectors(self):
        base = tiny_model(torch.bfloat16).base_model
        for block in base.blocks:
            del block.cam_encoder
            del block.projector
        config = {"offload_dtype": torch.bfloat16, "offload_device": "cpu",
                  "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                  "preparing_dtype": torch.bfloat16, "preparing_device": "cpu",
                  "computation_dtype": torch.bfloat16, "computation_device": "cpu"}
        from diffsynth.core import AutoWrappedNonRecurseModule
        base = enable_vram_management(base, {DiTBlock: AutoWrappedNonRecurseModule,
            Head: AutoWrappedModule, torch.nn.Linear: AutoWrappedLinear,
            torch.nn.Conv3d: AutoWrappedModule, torch.nn.LayerNorm: AutoWrappedModule,
            RMSNorm: AutoWrappedModule}, config)
        model = Manifold4DModel(base, positional_embedding_offset=4)
        self.assertIsInstance(base.blocks[0].cam_encoder, AutoWrappedLinear)
        self.assertIsInstance(base.blocks[0].projector, AutoWrappedLinear)
        self.assertTrue(torch.isfinite(self.forward(model)).all())
        with TemporaryDirectory() as folder:
            model.save_checkpoint(folder)
            model.load_checkpoint(folder)

    def test_two_stream_rope_and_invalid_grid(self):
        base = self.model.base_model
        freqs = two_stream_freqs(base, (2, 2, 2), 4, "cpu")
        torch.testing.assert_close(freqs[:4, 0, :2], base.freqs[0][0].expand(4, -1))
        torch.testing.assert_close(freqs[8:12, 0, :2], base.freqs[0][4].expand(4, -1))
        with self.assertRaises(ValueError):
            two_stream_freqs(base, (5, 2, 2), 4, "cpu")


class CameraTest(unittest.TestCase):
    def test_ray_centres_temporal_sampling_and_source_target_coordinates(self):
        c2w = torch.eye(4).expand(1, 9, 4, 4).clone()
        c2w[0, :, 0, 3] = torch.arange(9)
        intr = torch.tensor([16., 16., 16., 16.]).expand(1, 9, 4)
        rays = camera_tokens(c2w, intr, (3, 2, 2), (32, 32)).reshape(1, 3, 2, 2, 6)
        d = torch.tensor([-.5, -.5, 1.])
        d /= d.norm()
        torch.testing.assert_close(rays[0, 0, 0, 0, 3:], d)
        torch.testing.assert_close(rays[0, 0, ..., :3], torch.zeros(2, 2, 3))
        torch.testing.assert_close(rays[0, 1, 0, 0, :3], torch.cross(torch.tensor([4., 0., 0.]), d, dim=0))
        torch.testing.assert_close(rays[0, 2, 0, 0, :3], torch.cross(torch.tensor([8., 0., 0.]), d, dim=0))

    def test_intrinsics_follow_exact_integer_centre_crop(self):
        intr = np.array([[20., 30., 16., 19.]], np.float32)
        actual = resize_camera_intrinsics(intr, (39, 32), (32, 32))
        np.testing.assert_allclose(actual, [[20., 30., 16., 16.]])
        np.testing.assert_array_equal(intr, [[20., 30., 16., 19.]])


class SamplingTest(unittest.TestCase):
    def test_cfg_retains_geometry_and_sequential_matches_merged(self):
        class Model:
            def __init__(self):
                self.calls = []
            def __call__(self, latent, timestep, context, **conditions):
                self.calls.append((timestep.clone(), {k: v.clone() for k, v in conditions.items()}))
                return torch.ones_like(latent) * context.mean((1, 2))[:, None, None, None, None]
        render, noise = torch.ones(2, 16, 2, 4, 4), torch.randn(2, 16, 2, 4, 4)
        alpha = torch.rand(2, 1, 2, 4, 4)
        conditions = {"source_latents": render.clone(), "render_mask": alpha.repeat(1, 2, 1, 1, 1)}
        context, negative = torch.ones(2, 3, 8), torch.zeros(2, 3, 8)
        sequential = Model()
        result = sample_manifold(sequential, render, alpha, noise, context, conditions,
                                 negative_context=negative, solver="euler", num_steps=3)
        merged = Model()
        merged_result = sample_manifold(merged, render, alpha, noise, context, conditions,
                                        negative_context=negative, solver="euler", num_steps=3, cfg_merge=True)
        torch.testing.assert_close(result, merged_result)
        self.assertEqual(len(sequential.calls), 6)
        self.assertEqual(len(merged.calls), 3)
        for _, cond in sequential.calls:
            torch.testing.assert_close(cond["source_latents"], render)
            torch.testing.assert_close(cond["render_mask"], conditions["render_mask"])
            self.assertNotIn("render_latents", cond)
        torch.testing.assert_close(result, manifold_prior(render, alpha, noise) - 5.)

    def test_persistent_render_or_missing_negative_prompt_rejected(self):
        latent = torch.zeros(1, 16, 1, 2, 2)
        alpha = torch.ones(1, 1, 1, 2, 2)
        with self.assertRaises(ValueError):
            sample_manifold(None, latent, alpha, latent, torch.zeros(1, 2, 3), {})
        with self.assertRaises(ValueError):
            sample_manifold(None, latent, alpha, latent, torch.zeros(1, 2, 3), {"render": latent}, cfg_scale=1)


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoded = 0
    def encode(self, pixels, device, tiled=False):
        self.encoded += 1
        b, _, t, h, w = pixels.shape
        return torch.zeros(b, 16, (t - 1) // 4 + 1, h // 8, w // 8)
    def decode(self, latent, device, tiled=False):
        b, _, t, h, w = latent.shape
        return torch.zeros(b, 3, 4 * (t - 1) + 1, h * 8, w * 8)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.pipe = Manifold4DPipeline(device="cpu", torch_dtype=torch.float32)
        self.pipe.vae = FakeVAE()
        self.inputs = dict(
            source_video=[Image.new("RGB", (32, 32)) for _ in range(9)],
            point_cloud_video=[Image.new("RGB", (32, 32)) for _ in range(9)],
            point_cloud_alpha_mask=np.ones((9, 32, 32), np.float32),
            source_cam_c2w=np.tile(np.eye(4), (9, 1, 1)),
            target_cam_c2w=np.tile(np.eye(4), (9, 1, 1)),
            source_intrinsics=np.tile([16, 16, 16, 16], (9, 1)),
            target_intrinsics=np.tile([16, 16, 16, 16], (9, 1)),
            height=32, width=32, num_frames=9,
        )
        self.inputs["target_cam_c2w"][:, 0, 3] = 2

    def test_encode_render_once_and_keep_source_rays_separate(self):
        prepared = self.pipe.prepare_inputs(**self.inputs)
        self.assertEqual(self.pipe.vae.encoded, 2)
        self.assertEqual(prepared["source_latents"].shape, (1, 16, 3, 4, 4))
        self.assertEqual(prepared["source_mask"].shape, (1, 2, 3, 4, 4))
        rays = prepared["camera_embedding"]
        self.assertFalse(torch.allclose(rays[:, :12], rays[:, 12:]))
        self.assertEqual(prepared["source_mask"][:, 1:].count_nonzero().item(), 0)

    def test_alignment_and_shapes_fail_before_vae_encoding(self):
        with self.assertRaises(ValueError):
            self.pipe.prepare_inputs(**{**self.inputs, "target_cam_c2w": self.inputs["target_cam_c2w"][:5]})
        with self.assertRaises(ValueError):
            self.pipe.prepare_inputs(**{**self.inputs, "num_frames": 8})
        self.assertEqual(self.pipe.vae.encoded, 0)

    def test_inference_rejects_untrained_model(self):
        with self.assertRaisesRegex(ValueError, "trained Manifold4D"):
            self.pipe(prompt="test", **self.inputs)

    def test_full_pipeline_multiseed_sampling_decoding(self):
        calls = []
        class Model(torch.nn.Module):
            def forward(self, latent, timestep, context, **conditions):
                calls.append((latent.clone(), conditions))
                return torch.zeros_like(latent)
        self.pipe.dit = Model()
        self.pipe.checkpoint_loaded = True
        self.pipe.encode_prompt = lambda prompts: torch.zeros(len(prompts), 3, 8)
        clips = self.pipe(prompt="test", seed=[7, 7], solver="euler", num_inference_steps=2,
                          cfg_scale=1, rand_device="cpu", progress_bar_cmd=lambda x: x, **self.inputs)
        self.assertEqual(len(clips), 2)
        self.assertEqual(len(clips[0]), 9)
        self.assertEqual(clips[0][0].size, (32, 32))
        self.assertEqual(self.pipe.vae.encoded, 2)
        torch.testing.assert_close(calls[0][0][0], calls[0][0][1], rtol=0, atol=0)
        self.assertNotIn("render_latents", calls[0][1])

    def test_prepared_teacher_cache_validation(self):
        prepared = self.pipe.prepare_inputs(**self.inputs)
        prepared.update(target_latents=prepared["source_latents"].clone(),
                        context=torch.zeros(1, 512, 4096), empty_context=torch.zeros(1, 512, 4096))
        validate_training_example(prepared)
        prepared["target_latents"] = torch.zeros(1, 48, 3, 4, 4)
        with self.assertRaisesRegex(ValueError, "16"):
            validate_training_example(prepared)


if __name__ == "__main__":
    unittest.main()
