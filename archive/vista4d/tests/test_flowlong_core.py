import unittest

import torch

from diffsynth.pipelines.flowlong import (
    FlowLongSamplingConfig,
    aggregate_window_values,
    build_geometry_from_manifest,
    flowlong_next_state,
    overlap_error,
    run_flowlong_denoising,
    slice_global_latents,
)


def manifest_130(overlap=25, stride=24):
    starts = [0, 24, 48, 72, 96]
    clips = []
    for index, start in enumerate(starts):
        valid = min(49, 130 - start)
        clips.append(
            {
                "clip_index": index,
                "start_frame": start,
                "end_frame": start + valid - 1,
                "num_frames": 49,
                "valid_num_frames": valid,
                "pad_right": 49 - valid,
                "padded_end_frame": start + 48,
            }
        )
    return {
        "manifest_version": 2,
        "start_frame": 0,
        "end_exclusive": 130,
        "total_frames": 130,
        "clip_frames": 49,
        "overlap": overlap,
        "stride": stride,
        "temporal_alignment": 4,
        "tail_mode": "regular_stride_edge_padding",
        "padding_mode": "edge",
        "clips": clips,
    }


class FlowLongCoreTest(unittest.TestCase):
    def setUp(self):
        self.geometry = build_geometry_from_manifest(manifest_130())

    def test_geometry_for_current_video(self):
        geometry = self.geometry
        self.assertEqual(geometry.pixel_starts, (0, 24, 48, 72, 96))
        self.assertEqual(geometry.latent_starts, (0, 6, 12, 18, 24))
        self.assertEqual(
            (geometry.latent_window, geometry.latent_stride, geometry.latent_overlap),
            (13, 6, 7),
        )
        self.assertEqual(geometry.global_latent_frames, 37)
        self.assertEqual(geometry.padded_pixel_frames, 145)
        self.assertEqual(geometry.valid_pixel_frames, 130)
        self.assertEqual(geometry.trim_right, 15)

    def test_five_overlap_manifest_is_rejected(self):
        manifest = manifest_130(overlap=5, stride=44)
        with self.assertRaisesRegex(ValueError, "Expected overlap=25"):
            build_geometry_from_manifest(manifest)

    def test_slice_and_aggregate_round_trip_shared_global_values(self):
        global_values = torch.arange(37, dtype=torch.float32).reshape(1, 1, 37, 1, 1)
        windows = slice_global_latents(global_values, self.geometry)
        reconstructed = aggregate_window_values(windows, self.geometry)
        torch.testing.assert_close(reconstructed, global_values, rtol=0.0, atol=0.0)
        self.assertEqual(overlap_error(windows, self.geometry)["max_abs"], 0.0)

    def test_aggregation_uses_linear_blend_and_rightmost_pair_wins(self):
        windows = torch.stack(
            [torch.full((1, 13, 1, 1), float(index)) for index in range(5)],
            dim=0,
        )
        result = aggregate_window_values(windows, self.geometry)
        values = result[0, 0, :, 0, 0]

        self.assertEqual(values[0].item(), 0.0)
        self.assertEqual(values[6].item(), 0.0)
        self.assertAlmostEqual(values[9].item(), 0.5)
        self.assertEqual(values[12].item(), 1.0)
        self.assertEqual(values[18].item(), 2.0)
        self.assertEqual(values[24].item(), 3.0)
        self.assertEqual(values[30].item(), 4.0)
        self.assertEqual(values[36].item(), 4.0)

    def test_deterministic_next_state_matches_rectified_flow_line(self):
        x0 = torch.full((1, 1, 4, 1, 1), 2.0)
        x1 = torch.full_like(x0, 10.0)
        t = 0.8
        s = 0.3
        xt = (1.0 - t) * x0 + t * x1
        expected = (1.0 - s) * x0 + s * x1
        actual = flowlong_next_state(
            xt,
            x0,
            t=t,
            s=s,
            stochastic=False,
        )
        torch.testing.assert_close(actual, expected)

    def test_stochastic_next_state_is_reproducible(self):
        xt = torch.zeros((1, 1, 4, 1, 1))
        x0 = torch.ones_like(xt)
        first = flowlong_next_state(
            xt,
            x0,
            t=0.9,
            s=0.7,
            stochastic=True,
            generator=torch.Generator("cpu").manual_seed(123),
        )
        second = flowlong_next_state(
            xt,
            x0,
            t=0.9,
            s=0.7,
            stochastic=True,
            generator=torch.Generator("cpu").manual_seed(123),
        )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_joint_loop_finishes_all_windows_before_next_timestep(self):
        initial = torch.stack(
            [torch.full((1, 13, 1, 1), float(index)) for index in range(5)],
            dim=0,
        )
        calls = []

        def predict(step, windows):
            calls.append((step, windows.shape[0], overlap_error(windows, self.geometry)))
            return torch.zeros_like(windows)

        final, reports = run_flowlong_denoising(
            initial_window_latents=initial,
            geometry=self.geometry,
            sigmas=torch.tensor([1.0, 0.5]),
            predict_velocity=predict,
            config=FlowLongSamplingConfig(stochastic_enabled=False),
            stochastic_generator=None,
        )

        self.assertEqual([call[:2] for call in calls], [(0, 5), (1, 5)])
        self.assertGreater(calls[0][2]["max_abs"], 0.0)
        self.assertEqual(calls[1][2]["max_abs"], 0.0)
        self.assertEqual(final.shape, (1, 1, 37, 1, 1))
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report["overlap_after_max_abs"] == 0.0 for report in reports))

    def test_sampling_config_validation(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            FlowLongSamplingConfig(stochastic_threshold=0.0)
        with self.assertRaisesRegex(ValueError, "microbatch"):
            FlowLongSamplingConfig(microbatch_size=0)
        FlowLongSamplingConfig(stochastic_enabled=False, stochastic_threshold=0.0)


if __name__ == "__main__":
    unittest.main()
