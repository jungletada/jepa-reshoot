from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import numpy as np

from utils.media import (
    load_cameras,
    load_depths,
    load_masks,
    load_video,
    save_cameras,
    save_clips,
    save_depths,
    save_masks,
    save_video,
)
from utils.resolution import get_resolution_profile, validate_manifest_resolution
from utils.split_manifest import (
    clip_num_frames,
    clip_pad_right,
    clip_valid_num_frames,
    pad_first_axis_edge,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> Tuple[Dict, List[Dict], int, int]:
    if path.suffix.lower() != ".json":
        raise ValueError(
            "Full-condition slicing requires the JSON split manifest because it records "
            "the processed source range."
        )
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    clips = manifest.get("clips", [])
    if not clips:
        raise ValueError(f"No clips found in manifest: {path}")

    full_start = int(manifest.get("start_frame", min(int(clip["start_frame"]) for clip in clips)))
    full_end = int(manifest.get("end_exclusive", max(int(clip["end_frame"]) for clip in clips) + 1))
    if full_end <= full_start:
        raise ValueError(f"Invalid full range [{full_start}, {full_end}) in {path}")
    return manifest, clips, full_start, full_end


def derive_example_name(split_video_path: str, resolution: str, num_frames: int) -> str:
    stem = Path(split_video_path).stem
    for source_resolution in ("384p", "720p"):
        suffix = f"_{source_resolution}{num_frames}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + f"_{resolution}{num_frames}"
    return stem


def validate_lengths(expected: int, arrays: Dict[str, np.ndarray]) -> None:
    mismatches = {
        name: int(value.shape[0])
        for name, value in arrays.items()
        if value.shape[0] != expected
    }
    if mismatches:
        details = ", ".join(f"{name}={length}" for name, length in mismatches.items())
        raise ValueError(f"Expected {expected} full-sequence frames, got {details}")


def prepare_output_folder(output_folder: Path, overwrite: bool) -> bool:
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        return True

    metadata_path = output_folder / "full_sequence_slice.json"
    if not overwrite:
        if metadata_path.is_file():
            print(f"Skip existing full-sequence slice: {output_folder}")
            return False
        raise FileExistsError(
            f"Existing per-clip reconstruction was not created by the full-sequence workflow: {output_folder}. "
            "Pass --overwrite to replace only this recon_and_seg folder."
        )

    shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True)
    return True


def save_condition_video(path: Path, video: np.ndarray, fps: float, quality: int, lossless: bool) -> None:
    if lossless:
        save_video(
            str(path),
            video,
            fps=fps,
            quality=None,
            imageio_params={
                "codec": "libx264",
                "pixelformat": "yuv444p",
                "ffmpeg_params": ["-crf", "0", "-preset", "medium"],
            },
        )
    else:
        save_video(str(path), video, fps=fps, quality=quality)


def main(args) -> None:
    root = repo_root()
    manifest_path = resolve_path(args.manifest, root)
    full_folder = resolve_path(args.full_recon_and_seg_folder, root)
    result_root = resolve_path(args.result_root, root)
    smoothed_camera_path = (
        resolve_path(args.smoothed_camera, root)
        if args.smoothed_camera
        else full_folder / args.smoothed_camera_name
    )

    manifest, clips, full_start, full_end = load_manifest(manifest_path)
    profile = get_resolution_profile(args.resolution)
    validate_manifest_resolution(
        manifest,
        resolution=args.resolution,
        height=profile.height,
        width=profile.width,
    )
    full_frames = full_end - full_start

    video, fps = load_video(str(full_folder / "video.mp4"), desc="Loading full video")
    depths = load_depths(str(full_folder / "depths"), dtype=np.float16, desc="Loading full depths")
    dynamic_mask = load_masks(str(full_folder / "dynamic_mask"), desc="Loading full dynamic masks")
    sky_mask = load_masks(str(full_folder / "sky_mask"), desc="Loading full sky masks")
    raw_c2w, raw_intrinsics = load_cameras(str(full_folder / "cameras.npz"))
    smooth_c2w, smooth_intrinsics = load_cameras(str(smoothed_camera_path))

    validate_lengths(
        full_frames,
        {
            "video": video,
            "depths": depths,
            "dynamic_mask": dynamic_mask,
            "sky_mask": sky_mask,
            "raw_c2w": raw_c2w,
            "raw_intrinsics": raw_intrinsics,
            "smooth_c2w": smooth_c2w,
            "smooth_intrinsics": smooth_intrinsics,
        },
    )
    for name, value in (
        ("video", video),
        ("depths", depths),
        ("dynamic_mask", dynamic_mask),
        ("sky_mask", sky_mask),
    ):
        if tuple(value.shape[1:3]) != (profile.height, profile.width):
            raise ValueError(
                f"Full reconstruction {name} has spatial shape {value.shape[1:3]}, "
                f"expected {(profile.height, profile.width)} for {args.resolution}"
            )

    report = {
        "manifest": str(manifest_path),
        "full_recon_and_seg_folder": str(full_folder),
        "smoothed_camera": str(smoothed_camera_path),
        "full_start_frame": full_start,
        "full_end_exclusive": full_end,
        "full_frames": full_frames,
        "resolution": args.resolution,
        "height": profile.height,
        "width": profile.width,
        "num_frames": args.num_frames,
        "lossless_video": args.lossless_video,
        "clips": [],
    }

    for clip in sorted(clips, key=lambda value: int(value["clip_index"])):
        global_start = int(clip["start_frame"])
        global_end = int(clip["end_frame"])
        encoded_frames = clip_num_frames(clip)
        valid_frames = clip_valid_num_frames(clip)
        pad_right = clip_pad_right(clip)
        if encoded_frames != args.num_frames:
            raise ValueError(
                f"Clip {clip['clip_index']} encodes {encoded_frames} frames, "
                f"expected {args.num_frames}"
            )

        local_start = global_start - full_start
        local_end = global_end - full_start + 1
        if local_start < 0 or local_end > full_frames:
            raise ValueError(
                f"Clip {clip['clip_index']} range [{global_start}, {global_end}] is outside "
                f"the full processed range [{full_start}, {full_end})"
            )

        example = derive_example_name(clip["output_path"], args.resolution, args.num_frames)
        output_folder = result_root / example / "recon_and_seg"
        if not prepare_output_folder(output_folder, overwrite=args.overwrite):
            continue

        frame_slice = slice(local_start, local_end)
        save_condition_video(
            output_folder / "video.mp4",
            pad_first_axis_edge(video[frame_slice], encoded_frames),
            fps=fps,
            quality=args.quality,
            lossless=args.lossless_video,
        )
        save_depths(
            str(output_folder / "depths"),
            pad_first_axis_edge(depths[frame_slice], encoded_frames),
            dtype=np.float16,
        )
        save_masks(
            str(output_folder / "dynamic_mask"),
            pad_first_axis_edge(dynamic_mask[frame_slice], encoded_frames),
        )
        save_masks(
            str(output_folder / "sky_mask"),
            pad_first_axis_edge(sky_mask[frame_slice], encoded_frames),
        )
        save_cameras(
            str(output_folder / "cameras.npz"),
            pad_first_axis_edge(raw_c2w[frame_slice], encoded_frames),
            pad_first_axis_edge(raw_intrinsics[frame_slice], encoded_frames),
        )
        save_cameras(
            str(output_folder / args.smoothed_camera_name),
            pad_first_axis_edge(smooth_c2w[frame_slice], encoded_frames),
            pad_first_axis_edge(smooth_intrinsics[frame_slice], encoded_frames),
        )
        save_clips(str(output_folder / "clips.json"), {"src": (0, encoded_frames)})

        metadata = {
            "source": "full_sequence_recon_and_seg",
            "manifest": str(manifest_path),
            "full_recon_and_seg_folder": str(full_folder),
            "smoothed_camera": str(smoothed_camera_path),
            "clip_index": int(clip["clip_index"]),
            "global_start_frame": global_start,
            "global_end_frame": global_end,
            "full_local_start": local_start,
            "full_local_end_exclusive": local_end,
            "frames": encoded_frames,
            "valid_frames": valid_frames,
            "pad_right": pad_right,
            "padding_mode": "edge",
            "camera_coordinate_system": "shared_full_sequence_world",
            "lossless_video": args.lossless_video,
        }
        with (output_folder / "full_sequence_slice.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        report["clips"].append({"example": example, "output_folder": str(output_folder), **metadata})
        print(
            f"[{int(clip['clip_index']):03d}] global frames {global_start}..{global_end} "
            f"({valid_frames} valid, pad_right={pad_right}) -> {output_folder}"
        )

    report_path = result_root / f"{manifest_path.stem}_full_sequence_conditions.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Condition slicing report: {report_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Slice one full-sequence Pi3/SAM3 reconstruction into per-clip Vista4D conditions."
    )
    parser.add_argument("--manifest", required=True, help="JSON manifest created by split_video_into_clips.py")
    parser.add_argument("--full_recon_and_seg_folder", required=True)
    parser.add_argument("--smoothed_camera", default=None)
    parser.add_argument("--smoothed_camera_name", default="cameras_gaussian_smooth.npz")
    parser.add_argument("--result_root", default="./results/single")
    parser.add_argument("--resolution", default="384p")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--quality", type=int, default=9)
    parser.add_argument("--lossy_video", dest="lossless_video", action="store_false")
    parser.set_defaults(lossless_video=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    main(parser.parse_args())
