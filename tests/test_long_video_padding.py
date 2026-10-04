import unittest
from pathlib import Path

import numpy as np

from scripts.postprocess.merge_split_videos import (
    merge_videos,
    prepare_clip_video,
)
from scripts.preprocess.split_video_into_clips import build_clip_starts
from utils.split_manifest import (
    clip_num_frames,
    clip_pad_right,
    clip_valid_num_frames,
    pad_first_axis_edge,
)


class LongVideoPaddingTest(unittest.TestCase):
    def test_flowlong_130_frames_use_25_overlap_and_24_stride(self):
        starts = build_clip_starts(
            start_frame=0,
            end_exclusive=130,
            clip_frames=49,
            overlap=25,
            include_tail=True,
            temporal_alignment=4,
        )

        self.assertEqual(starts, [0, 24, 48, 72, 96])
        self.assertEqual(np.diff(starts).tolist(), [24] * 4)
        self.assertTrue(all(start % 4 == 0 for start in starts))

        valid_num_frames = [min(49, 130 - start) for start in starts]
        pad_right = [49 - valid for valid in valid_num_frames]
        self.assertEqual(valid_num_frames, [49, 49, 49, 49, 34])
        self.assertEqual(pad_right, [0, 0, 0, 0, 15])

        latent_window = (49 - 1) // 4 + 1
        latent_stride = (49 - 25) // 4
        latent_overlap = latent_window - latent_stride
        self.assertEqual(
            (latent_window, latent_stride, latent_overlap),
            (13, 6, 7),
        )

    def test_130_frames_use_regular_aligned_starts(self):
        starts = build_clip_starts(
            start_frame=0,
            end_exclusive=130,
            clip_frames=49,
            overlap=5,
            include_tail=True,
            temporal_alignment=4,
        )
        self.assertEqual(starts, [0, 44, 88])
        self.assertTrue(all(start % 4 == 0 for start in starts))
        self.assertEqual(starts[-1] + 49 - 130, 7)

    def test_310_frames_keep_stride_and_pad_tail(self):
        starts = build_clip_starts(
            start_frame=0,
            end_exclusive=310,
            clip_frames=49,
            overlap=5,
            include_tail=True,
            temporal_alignment=4,
        )
        self.assertEqual(starts, [0, 44, 88, 132, 176, 220, 264])
        self.assertEqual(np.diff(starts).tolist(), [44] * 6)
        self.assertEqual(starts[-1] + 49 - 310, 3)

    def test_invalid_temporal_stride_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stride=.*must be divisible"):
            build_clip_starts(
                start_frame=0,
                end_exclusive=130,
                clip_frames=49,
                overlap=4,
                include_tail=True,
                temporal_alignment=4,
            )

    def test_manifest_padding_is_backward_compatible(self):
        legacy = {
            "start_frame": 44,
            "end_frame": 92,
            "num_frames": 49,
        }
        padded = {
            "start_frame": 88,
            "end_frame": 129,
            "num_frames": 49,
            "valid_num_frames": 42,
            "pad_right": 7,
            "padded_end_frame": 136,
        }
        self.assertEqual(clip_num_frames(legacy), 49)
        self.assertEqual(clip_valid_num_frames(legacy), 49)
        self.assertEqual(clip_pad_right(legacy), 0)
        self.assertEqual(clip_num_frames(padded), 49)
        self.assertEqual(clip_valid_num_frames(padded), 42)
        self.assertEqual(clip_pad_right(padded), 7)

    def test_edge_padding_repeats_last_real_frame(self):
        value = np.arange(6, dtype=np.int64).reshape(3, 2)
        padded = pad_first_axis_edge(value, 5)
        self.assertEqual(padded.shape, (5, 2))
        np.testing.assert_array_equal(padded[:3], value)
        np.testing.assert_array_equal(padded[3], value[-1])
        np.testing.assert_array_equal(padded[4], value[-1])

    def test_merge_trims_padded_generated_frames(self):
        clips = [
            {
                "clip_index": 0,
                "start_frame": 0,
                "end_frame": 48,
                "num_frames": 49,
            },
            {
                "clip_index": 1,
                "start_frame": 44,
                "end_frame": 92,
                "num_frames": 49,
            },
            {
                "clip_index": 2,
                "start_frame": 88,
                "end_frame": 129,
                "num_frames": 49,
                "valid_num_frames": 42,
                "pad_right": 7,
                "padded_end_frame": 136,
            },
        ]
        videos = []
        for clip in clips:
            valid = clip_valid_num_frames(clip)
            values = np.arange(
                clip["start_frame"],
                clip["start_frame"] + valid,
                dtype=np.uint8,
            ).reshape(valid, 1, 1, 1)
            values = np.repeat(values, 3, axis=3)
            encoded = pad_first_axis_edge(values, clip_num_frames(clip))
            videos.append(
                prepare_clip_video(
                    encoded,
                    clip,
                    strict_num_frames=True,
                    path=Path(f"clip_{clip['clip_index']}.mp4"),
                )
            )

        merged, _ = merge_videos(
            clips,
            videos,
            mode="center_cut",
            flow_max_pixels=32.0,
            flow_consistency_sigma=2.5,
        )
        self.assertEqual(merged.shape[0], 130)
        np.testing.assert_array_equal(
            merged[:, 0, 0, 0],
            np.arange(130, dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
