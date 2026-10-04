from typing import Any, Dict, List, Optional

import numpy as np


def clip_num_frames(clip: Dict[str, Any], default: Optional[int] = None) -> int:
    """Return the encoded frame count for a split, including right padding."""
    value = clip.get("num_frames", default)
    if value is None:
        value = int(clip["end_frame"]) - int(clip["start_frame"]) + 1
    value = int(value)
    if value <= 0:
        raise ValueError(f"Invalid clip num_frames={value}: {clip}")
    return value


def clip_valid_num_frames(clip: Dict[str, Any]) -> int:
    """Return the number of frames backed by real source frames."""
    encoded = clip_num_frames(clip)
    inferred = int(clip["end_frame"]) - int(clip["start_frame"]) + 1
    value = int(clip.get("valid_num_frames", inferred))
    if value <= 0 or value > encoded:
        raise ValueError(
            f"Invalid valid_num_frames={value} for encoded num_frames={encoded}: {clip}"
        )
    if inferred != value:
        raise ValueError(
            f"Manifest interval contains {inferred} real frames but "
            f"valid_num_frames={value}: {clip}"
        )
    return value


def clip_pad_right(clip: Dict[str, Any]) -> int:
    encoded = clip_num_frames(clip)
    valid = clip_valid_num_frames(clip)
    inferred = encoded - valid
    value = int(clip.get("pad_right", inferred))
    if value != inferred:
        raise ValueError(
            f"Invalid pad_right={value}; expected {inferred} from "
            f"num_frames={encoded}, valid_num_frames={valid}: {clip}"
        )
    return value


def pad_first_axis_edge(value: np.ndarray, target_frames: int) -> np.ndarray:
    """Right-pad an array by repeating its final frame."""
    if value.ndim == 0:
        raise ValueError("Cannot frame-pad a scalar array")
    current_frames = int(value.shape[0])
    if current_frames <= 0:
        raise ValueError("Cannot frame-pad an empty array")
    if current_frames > target_frames:
        raise ValueError(
            f"Array has {current_frames} frames, exceeding target {target_frames}"
        )
    if current_frames == target_frames:
        return value
    padding = np.repeat(value[-1:], target_frames - current_frames, axis=0)
    return np.concatenate((value, padding), axis=0)


def validate_flowlong_window_manifest(
    manifest: Dict[str, Any],
    clip_frames: int = 49,
    overlap: int = 25,
    temporal_alignment: int = 4,
) -> List[Dict[str, Any]]:
    """Validate the regular pixel-window contract required by FlowLong."""
    if int(manifest.get("manifest_version", 0)) < 2:
        raise ValueError("FlowLong requires manifest_version >= 2")
    clips = sorted(
        manifest.get("clips", []), key=lambda clip: int(clip["clip_index"])
    )
    if len(clips) < 2:
        raise ValueError("FlowLong requires at least two windows")
    if clip_frames <= 0:
        raise ValueError(f"clip_frames must be positive, got {clip_frames}")
    if overlap < 0 or overlap >= clip_frames:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < clip_frames, got {overlap}"
        )
    if temporal_alignment <= 0:
        raise ValueError(
            f"temporal_alignment must be positive, got {temporal_alignment}"
        )

    stride = clip_frames - overlap
    expected_metadata = {
        "clip_frames": clip_frames,
        "overlap": overlap,
        "stride": stride,
        "temporal_alignment": temporal_alignment,
    }
    for name, expected in expected_metadata.items():
        actual = int(manifest.get(name, -1))
        if actual != expected:
            raise ValueError(f"Expected {name}={expected}, got {actual}")
    if stride % temporal_alignment:
        raise ValueError(
            f"stride={stride} must be divisible by temporal_alignment="
            f"{temporal_alignment}"
        )
    if (clip_frames - 1) % temporal_alignment:
        raise ValueError(
            f"clip_frames={clip_frames} must satisfy "
            "(clip_frames - 1) % temporal_alignment == 0"
        )

    starts = [int(clip["start_frame"]) for clip in clips]
    expected_starts = [starts[0] + index * stride for index in range(len(clips))]
    if starts != expected_starts:
        raise ValueError(f"Window starts are not a regular stride grid: {starts}")
    if any(start % temporal_alignment for start in starts):
        raise ValueError(
            f"Window starts are not aligned to {temporal_alignment}: {starts}"
        )
    manifest_start = int(manifest.get("start_frame", starts[0]))
    if manifest_start != starts[0]:
        raise ValueError(
            f"Manifest start_frame={manifest_start} does not match first window "
            f"start={starts[0]}"
        )

    for index, clip in enumerate(clips):
        clip_index = int(clip["clip_index"])
        if clip_index != index:
            raise ValueError("clip_index values must be contiguous and start at zero")
        if clip_num_frames(clip) != clip_frames:
            raise ValueError(f"Clip {clip_index} does not encode {clip_frames} frames")
        valid = clip_valid_num_frames(clip)
        padding = clip_pad_right(clip)
        if index < len(clips) - 1 and (valid != clip_frames or padding != 0):
            raise ValueError(
                f"Only the final clip may contain padding; clip {clip_index} is padded"
            )
        padded_end = int(clip.get("padded_end_frame", starts[index] + clip_frames - 1))
        if padded_end != starts[index] + clip_frames - 1:
            raise ValueError(
                f"Clip {clip_index} padded_end_frame={padded_end} is inconsistent"
            )
        if index:
            previous_end = int(clips[index - 1]["end_frame"])
            if previous_end + 1 < starts[index]:
                raise ValueError(f"Gap before clip {clip_index}")

    manifest_end = int(
        manifest.get("end_exclusive", int(clips[-1]["end_frame"]) + 1)
    )
    final_end = int(clips[-1]["end_frame"]) + 1
    if final_end != manifest_end:
        raise ValueError(
            f"Final real frame ends at {final_end}, but manifest end_exclusive="
            f"{manifest_end}"
        )
    if manifest.get("tail_mode") not in (None, "regular_stride_edge_padding"):
        raise ValueError(
            "FlowLong requires tail_mode=regular_stride_edge_padding, got "
            f"{manifest.get('tail_mode')}"
        )
    if manifest.get("padding_mode") not in (None, "edge"):
        raise ValueError(
            f"FlowLong requires padding_mode=edge, got {manifest.get('padding_mode')}"
        )
    return clips
