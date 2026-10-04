from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ResolutionProfile:
    name: str
    height: int
    width: int
    vista4d_checkpoint_folder: str


RESOLUTION_PROFILES = {
    "384p": ResolutionProfile(
        name="384p",
        height=384,
        width=672,
        vista4d_checkpoint_folder="384p49_step=30000",
    ),
    "720p": ResolutionProfile(
        name="720p",
        height=720,
        width=1280,
        vista4d_checkpoint_folder="720p49_step=3000",
    ),
}


def get_resolution_profile(resolution: str) -> ResolutionProfile:
    try:
        return RESOLUTION_PROFILES[resolution]
    except KeyError as error:
        supported = ", ".join(RESOLUTION_PROFILES)
        raise ValueError(
            f"Unsupported resolution={resolution!r}; expected one of: {supported}"
        ) from error


def validate_resolution_dimensions(
    resolution: str,
    *,
    height: int,
    width: int,
) -> ResolutionProfile:
    profile = get_resolution_profile(resolution)
    actual = (int(width), int(height))
    expected = (profile.width, profile.height)
    if actual != expected:
        raise ValueError(
            f"resolution={resolution} requires width x height "
            f"{profile.width}x{profile.height}, got {actual[0]}x{actual[1]}"
        )
    return profile


def validate_manifest_resolution(
    manifest: Dict[str, Any],
    *,
    resolution: str,
    height: int,
    width: int,
) -> None:
    """Reject a manifest created for a different resolution.

    Manifests written before resolution metadata was introduced remain valid when
    their existing target dimensions match. New manifests carry all three fields.
    """
    validate_resolution_dimensions(resolution, height=height, width=width)
    expected = {
        "resolution": resolution,
        "target_height": int(height),
        "target_width": int(width),
    }
    for key, expected_value in expected.items():
        actual = manifest.get(key)
        if actual is not None and actual != expected_value:
            raise ValueError(
                f"Manifest {key}={actual!r} does not match expected "
                f"{expected_value!r}"
            )
