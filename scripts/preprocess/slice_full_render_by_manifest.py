from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import imageio
import numpy as np

from utils.media import (
    load_cameras,
    load_depths,
    load_masks,
    load_video,
    save_cameras,
    save_depths,
    save_masks,
)
from utils.resolution import get_resolution_profile, validate_manifest_resolution
from utils.split_manifest import (
    clip_num_frames,
    clip_pad_right,
    clip_valid_num_frames,
    pad_first_axis_edge,
)


VIDEO_NAMES = ("video_src", "video_pc")
DEPTH_NAMES = ("depths_src", "depths_pc")
MASK_NAMES = (
    "alpha_mask_src",
    "alpha_mask_pc",
    "dynamic_mask_src",
    "dynamic_mask_pc",
    "static_mask_src",
    "static_mask_pc",
    "sky_mask_src",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> Tuple[Dict, List[Dict]]:
    if path.suffix.lower() != ".json":
        raise ValueError("Full render slicing requires the JSON split manifest")
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    clips = sorted(manifest.get("clips", []), key=lambda value: int(value["clip_index"]))
    if not clips:
        raise ValueError(f"No clips found in manifest: {path}")
    return manifest, clips


def derive_example_name(split_video_path: str, resolution: str, num_frames: int) -> str:
    stem = Path(split_video_path).stem
    for source_resolution in ("384p", "720p"):
        suffix = f"_{source_resolution}{num_frames}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + f"_{resolution}{num_frames}"
    return stem


def save_lossless_rgb_video(path: Path, video: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264rgb",
        input_params=["-r", format(float(fps), ".12g")],
        pixelformat="rgb24",
        ffmpeg_params=["-crf", "0", "-preset", "medium"],
        macro_block_size=1,
    )
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()


def prepare_output_folder(folder: Path, overwrite: bool) -> None:
    if folder.exists():
        if not overwrite:
            raise FileExistsError(
                f"Render folder already exists: {folder}. Pass --overwrite to replace it."
            )
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def validate_lengths(expected: int, arrays: Dict[str, np.ndarray]) -> None:
    mismatches = {
        name: int(value.shape[0])
        for name, value in arrays.items()
        if value.shape[0] != expected
    }
    if mismatches:
        raise ValueError(f"Full render length mismatch: {mismatches}; expected {expected}")


def main(args) -> None:
    root = repo_root()
    manifest_path = resolve_path(args.manifest, root)
    full_folder = resolve_path(args.full_render_folder, root)
    result_root = resolve_path(args.result_root, root)
    manifest, clips = load_manifest(manifest_path)
    profile = get_resolution_profile(args.resolution)
    validate_manifest_resolution(
        manifest,
        resolution=args.resolution,
        height=profile.height,
        width=profile.width,
    )

    metadata_path = full_folder / "shared_static_render.json"
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    metadata_size = (int(metadata.get("width", -1)), int(metadata.get("height", -1)))
    expected_size = (profile.width, profile.height)
    if metadata_size != expected_size:
        raise ValueError(
            f"{metadata_path} records {metadata_size[0]}x{metadata_size[1]}, "
            f"expected {profile.width}x{profile.height} for {args.resolution}"
        )
    full_start = int(metadata["global_start_frame"])
    full_end = int(metadata["global_end_exclusive"])
    full_frames = full_end - full_start

    arrays = {}
    fps_values = []
    for name in VIDEO_NAMES:
        arrays[name], fps = load_video(
            str(full_folder / f"{name}.mp4"), desc=f"Loading full {name}"
        )
        fps_values.append(float(fps))
    for name in DEPTH_NAMES:
        arrays[name] = load_depths(
            str(full_folder / name), dtype=np.float16, desc=f"Loading full {name}"
        )
    for name in MASK_NAMES:
        arrays[name] = load_masks(
            str(full_folder / name), desc=f"Loading full {name}"
        )
    cameras_src, intrinsics_src = load_cameras(str(full_folder / "cameras_src.npz"))
    cameras_tgt, intrinsics_tgt = load_cameras(str(full_folder / "cameras_tgt.npz"))
    arrays.update(
        {
            "cameras_src": cameras_src,
            "intrinsics_src": intrinsics_src,
            "cameras_tgt": cameras_tgt,
            "intrinsics_tgt": intrinsics_tgt,
        }
    )
    validate_lengths(full_frames, arrays)
    for name in VIDEO_NAMES + DEPTH_NAMES + MASK_NAMES:
        if tuple(arrays[name].shape[1:3]) != (profile.height, profile.width):
            raise ValueError(
                f"Full render {name} has spatial shape {arrays[name].shape[1:3]}, "
                f"expected {(profile.height, profile.width)}"
            )
    if max(fps_values) - min(fps_values) > 1e-3:
        raise ValueError(f"Full render video FPS mismatch: {fps_values}")
    fps = fps_values[0]

    report = {
        "source": "full_shared_static_render_slice",
        "manifest": str(manifest_path),
        "full_render_folder": str(full_folder),
        "result_root": str(result_root),
        "full_start_frame": full_start,
        "full_end_exclusive": full_end,
        "full_frames": full_frames,
        "resolution": args.resolution,
        "height": profile.height,
        "width": profile.width,
        "num_frames": args.num_frames,
        "clips": [],
    }
    for clip in clips:
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
                f"Clip {clip['clip_index']} range {global_start}..{global_end} is "
                f"outside full render range {full_start}..{full_end - 1}"
            )
        frame_slice = slice(local_start, local_end)
        example = derive_example_name(
            clip["output_path"], args.resolution, args.num_frames
        )
        output_folder = result_root / example / args.render_folder_name
        prepare_output_folder(output_folder, overwrite=args.overwrite)

        for name in VIDEO_NAMES:
            save_lossless_rgb_video(
                output_folder / f"{name}.mp4",
                pad_first_axis_edge(arrays[name][frame_slice], encoded_frames),
                fps=fps,
            )
        for name in DEPTH_NAMES:
            save_depths(
                str(output_folder / name),
                pad_first_axis_edge(arrays[name][frame_slice], encoded_frames),
                dtype=np.float16,
            )
        for name in MASK_NAMES:
            save_masks(
                str(output_folder / name),
                pad_first_axis_edge(arrays[name][frame_slice], encoded_frames),
            )
        save_cameras(
            str(output_folder / "cameras_src.npz"),
            pad_first_axis_edge(cameras_src[frame_slice], encoded_frames),
            pad_first_axis_edge(intrinsics_src[frame_slice], encoded_frames),
        )
        save_cameras(
            str(output_folder / "cameras_tgt.npz"),
            pad_first_axis_edge(cameras_tgt[frame_slice], encoded_frames),
            pad_first_axis_edge(intrinsics_tgt[frame_slice], encoded_frames),
        )
        clip_metadata = {
            "source": "full_shared_static_render_slice",
            "full_render_folder": str(full_folder),
            "clip_index": int(clip["clip_index"]),
            "global_start_frame": global_start,
            "global_end_frame": global_end,
            "full_local_start": local_start,
            "full_local_end_exclusive": local_end,
            "frames": encoded_frames,
            "valid_frames": valid_frames,
            "pad_right": pad_right,
            "padding_mode": "edge",
        }
        with (output_folder / "full_shared_static_slice.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(clip_metadata, file, indent=2)
        report["clips"].append(
            {"example": example, "output_folder": str(output_folder), **clip_metadata}
        )
        print(
            f"[{int(clip['clip_index']):03d}] global frames "
            f"{global_start}..{global_end} "
            f"({valid_frames} valid, pad_right={pad_right}) -> {output_folder}"
        )

    result_root.mkdir(parents=True, exist_ok=True)
    report_path = result_root / f"{manifest_path.stem}_shared_static_render.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Full shared-static render slicing report: {report_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Slice one full shared-static render into overlapping Vista4D windows."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--full_render_folder", required=True)
    parser.add_argument("--result_root", default="./results/shared_static_single")
    parser.add_argument("--render_folder_name", default="render_384p_smooth")
    parser.add_argument("--resolution", default="384p")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--overwrite", action="store_true", default=False)
    main(parser.parse_args())
