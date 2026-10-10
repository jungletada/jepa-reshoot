import unittest

from utils.resolution import (
    get_resolution_profile,
    validate_manifest_resolution,
    validate_resolution_dimensions,
)


class ResolutionProfilesTest(unittest.TestCase):
    def test_supported_profiles_have_locked_dimensions_and_checkpoints(self):
        profile_384 = get_resolution_profile("384p")
        self.assertEqual(
            (profile_384.width, profile_384.height),
            (672, 384),
        )
        self.assertEqual(
            profile_384.vista4d_checkpoint_folder,
            "384p49_step=30000",
        )

        profile_720 = get_resolution_profile("720p")
        self.assertEqual(
            (profile_720.width, profile_720.height),
            (1280, 720),
        )
        self.assertEqual(
            profile_720.vista4d_checkpoint_folder,
            "720p49_step=3000",
        )

    def test_unknown_or_mislabeled_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported resolution"):
            get_resolution_profile("1080p")
        with self.assertRaisesRegex(ValueError, "requires width x height"):
            validate_resolution_dimensions("720p", height=384, width=672)

    def test_manifest_resolution_is_checked_but_legacy_metadata_is_allowed(self):
        validate_manifest_resolution(
            {},
            resolution="384p",
            height=384,
            width=672,
        )
        validate_manifest_resolution(
            {
                "resolution": "720p",
                "target_height": 720,
                "target_width": 1280,
            },
            resolution="720p",
            height=720,
            width=1280,
        )
        with self.assertRaisesRegex(ValueError, "Manifest resolution"):
            validate_manifest_resolution(
                {
                    "resolution": "384p",
                    "target_height": 384,
                    "target_width": 672,
                },
                resolution="720p",
                height=720,
                width=1280,
            )


if __name__ == "__main__":
    unittest.main()
