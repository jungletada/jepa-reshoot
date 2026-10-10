from types import MethodType
import unittest

import torch

from diffsynth.pipelines.flowlong import (
    FlowLongSamplingConfig,
    build_geometry_from_manifest,
)
from diffsynth.pipelines.wan_video_vista4d import (
    Vista4DPipeline,
    _slice_batch_value,
    _slice_flowlong_batch,
)
from tests.test_flowlong_core import manifest_130


class FlowLongPipelineTest(unittest.TestCase):
    def test_existing_single_window_call_keeps_cfg_scheduler_path(self):
        pipe = Vista4DPipeline(device="cpu", torch_dtype=torch.float32)
        pipe.dit = object()
        pipe.dit2 = None
        calls = []

        def prepare(self, **kwargs):
            return (
                {"latents": torch.zeros((1, 1, 1, 1, 1))},
                {"context": torch.tensor([1.0])},
                {"context": torch.tensor([-1.0])},
            )

        def model_fn(**kwargs):
            calls.append(float(kwargs["context"].item()))
            return kwargs["latents"] + kwargs["context"].reshape(1, 1, 1, 1, 1)

        def decode(self, latents, **kwargs):
            return latents

        pipe._prepare_inference_inputs = MethodType(prepare, pipe)
        pipe._decode_latents = MethodType(decode, pipe)
        pipe.model_fn = model_fn
        pipe.load_models_to_device = lambda names: None

        output = pipe(
            prompt=["prompt"],
            negative_prompt=["negative"],
            seed=[1],
            cfg_scale=5.0,
            num_inference_steps=2,
            progress_bar_cmd=lambda values: values,
            output_type="floatpoint",
        )

        self.assertEqual(calls, [1.0, -1.0, 1.0, -1.0])
        self.assertEqual(output.shape, (1, 1, 1, 1, 1))
        self.assertLess(float(output.item()), 0.0)

    def test_explicit_batch_slicing_preserves_order(self):
        values = {
            "latents": torch.arange(5),
            "context": torch.arange(10).reshape(5, 2),
            "cam_emb": torch.arange(5),
            "not_batch_sensitive": torch.arange(7),
        }
        sliced = _slice_flowlong_batch(values, start=2, end=4, batch_size=5)
        torch.testing.assert_close(sliced["latents"], torch.tensor([2, 3]))
        torch.testing.assert_close(
            sliced["context"],
            torch.tensor([[4, 5], [6, 7]]),
        )
        torch.testing.assert_close(sliced["cam_emb"], torch.tensor([2, 3]))
        self.assertIs(sliced["not_batch_sensitive"], values["not_batch_sensitive"])

    def test_invalid_batch_sensitive_leading_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 1 or 5"):
            _slice_batch_value(
                torch.zeros(3, 2),
                start=0,
                end=1,
                batch_size=5,
                name="context",
            )

    def test_microbatch_cfg_matches_full_batch_formula(self):
        pipe = Vista4DPipeline(device="cpu", torch_dtype=torch.float32)

        def model_fn(**kwargs):
            context = kwargs["context"].reshape(-1, 1, 1, 1, 1)
            return kwargs["latents"] + context

        pipe.model_fn = model_fn
        latents = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1, 1)
        positive_context = torch.arange(5, dtype=torch.float32)
        negative_context = -positive_context
        actual = pipe._predict_flowlong_velocity(
            models={"dit": object()},
            inputs_shared={"latents": latents},
            inputs_posi={"context": positive_context},
            inputs_nega={"context": negative_context},
            window_latents=latents,
            timestep_value=torch.tensor(500.0),
            cfg_scale=5.0,
            microbatch_size=2,
        )
        positive = latents + positive_context.reshape(5, 1, 1, 1, 1)
        negative = latents + negative_context.reshape(5, 1, 1, 1, 1)
        expected = negative + 5.0 * (positive - negative)
        torch.testing.assert_close(actual, expected)

    def test_generate_flowlong_mock_runs_joint_steps_and_trims_decode(self):
        geometry = build_geometry_from_manifest(manifest_130())
        initial = torch.stack(
            [torch.full((1, 13, 1, 1), float(index)) for index in range(5)],
            dim=0,
        )
        pipe = Vista4DPipeline(device="cpu", torch_dtype=torch.float32)
        pipe.dit = object()
        pipe.dit2 = None
        model_calls = []
        model_loads = []

        def prepare(self, **kwargs):
            return {"latents": initial.clone()}, {}, {}

        def predict(self, **kwargs):
            windows = kwargs["window_latents"]
            model_calls.append(windows.clone())
            return torch.zeros_like(windows)

        def decode(self, latents, **kwargs):
            self.assert_global_shape = tuple(latents.shape)
            return torch.zeros((1, 3, 145, 1, 1), dtype=torch.float32)

        pipe._prepare_inference_inputs = MethodType(prepare, pipe)
        pipe._predict_flowlong_velocity = MethodType(predict, pipe)
        pipe._decode_latents = MethodType(decode, pipe)
        pipe.load_models_to_device = lambda names: model_loads.append(tuple(names))

        output, report = pipe.generate_flowlong(
            flowlong_geometry=geometry,
            flowlong_config=FlowLongSamplingConfig(stochastic_enabled=False),
            base_seed=10027,
            prompt=["same prompt"] * 5,
            negative_prompt=["same negative prompt"] * 5,
            num_inference_steps=2,
            output_type="floatpoint",
            progress_bar_cmd=lambda values: values,
        )

        self.assertEqual(len(model_calls), 2)
        self.assertEqual([call.shape[0] for call in model_calls], [5, 5])
        self.assertEqual(pipe.assert_global_shape, (1, 1, 37, 1, 1))
        self.assertEqual(output.shape, (1, 3, 130, 1, 1))
        self.assertEqual(report["window_seeds"], [10027, 10028, 10029, 10030, 10031])
        self.assertEqual(report["stochastic_seed"], 1010030)
        self.assertEqual(report["decoded_frames_before_trim"], 145)
        self.assertEqual(report["trimmed_frames"], 15)
        self.assertTrue(
            all(step["overlap_after_max_abs"] == 0.0 for step in report["steps"])
        )
        self.assertEqual(model_loads[-1], ())


if __name__ == "__main__":
    unittest.main()
