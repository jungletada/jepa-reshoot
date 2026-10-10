from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

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
from utils.split_manifest import clip_num_frames, clip_valid_num_frames


@dataclass
class ClipData:
    index: int
    global_start: int
    global_end: int
    folder: Path
    video: np.ndarray
    fps: float
    depths: np.ndarray
    dynamic_mask: np.ndarray
    sky_mask: np.ndarray
    c2w: np.ndarray
    intrinsics: np.ndarray


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def derive_example_name(split_video_path: str, resolution: str, num_frames: int) -> str:
    stem = Path(split_video_path).stem
    for source_resolution in ("384p", "720p"):
        suffix = f"_{source_resolution}{num_frames}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + f"_{resolution}{num_frames}"
    return stem


def load_manifest(path: Path) -> Tuple[Dict, List[Dict], int, int]:
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    clips = sorted(manifest.get("clips", []), key=lambda value: int(value["clip_index"]))
    if not clips:
        raise ValueError(f"No clips found in manifest: {path}")

    full_start = int(manifest.get("start_frame", min(int(clip["start_frame"]) for clip in clips)))
    full_end = int(manifest.get("end_exclusive", max(int(clip["end_frame"]) for clip in clips) + 1))
    if full_end <= full_start:
        raise ValueError(f"Invalid full range [{full_start}, {full_end}) in {path}")
    return manifest, clips, full_start, full_end


def load_clip(
    clip: Dict,
    input_result_root: Path,
    resolution: str,
    num_frames: int,
) -> ClipData:
    index = int(clip["clip_index"])
    global_start = int(clip["start_frame"])
    global_end = int(clip["end_frame"])
    expected = clip_num_frames(clip)
    valid_frames = clip_valid_num_frames(clip)
    if expected != num_frames:
        raise ValueError(
            f"Clip {index} encodes {expected} frames, expected {num_frames}"
        )
    if global_end - global_start + 1 != valid_frames:
        raise ValueError(
            f"Clip {index} interval {global_start}..{global_end} does not match "
            f"valid_num_frames={valid_frames}"
        )

    example = derive_example_name(clip["output_path"], resolution, num_frames)
    folder = input_result_root / example / "recon_and_seg"
    required = [
        folder / "video.mp4",
        folder / "depths",
        folder / "dynamic_mask",
        folder / "sky_mask",
        folder / "cameras.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reconstruction inputs for clip {index}: {missing}")

    video, fps = load_video(str(folder / "video.mp4"), desc=f"Loading clip {index} video")
    depths = load_depths(
        str(folder / "depths"),
        dtype=np.float32,
        desc=f"Loading clip {index} depths",
    )
    dynamic_mask = load_masks(
        str(folder / "dynamic_mask"),
        desc=f"Loading clip {index} dynamic masks",
    )
    sky_mask = load_masks(
        str(folder / "sky_mask"),
        desc=f"Loading clip {index} sky masks",
    )
    c2w, intrinsics = load_cameras(str(folder / "cameras.npz"))

    lengths = {
        "video": len(video),
        "depths": len(depths),
        "dynamic_mask": len(dynamic_mask),
        "sky_mask": len(sky_mask),
        "c2w": len(c2w),
        "intrinsics": len(intrinsics),
    }
    mismatches = {name: length for name, length in lengths.items() if length != expected}
    if mismatches:
        raise ValueError(f"Clip {index} expected {expected} frames, got {mismatches}")
    if c2w.shape[1:] != (4, 4):
        raise ValueError(f"Clip {index} has invalid C2W shape: {c2w.shape}")

    c2w = np.asarray(c2w, dtype=np.float64)
    c2w[:, :3, :3] = Rotation.from_matrix(c2w[:, :3, :3]).as_matrix()
    c2w[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])

    return ClipData(
        index=index,
        global_start=global_start,
        global_end=global_end,
        folder=folder,
        video=video,
        fps=float(fps),
        depths=np.asarray(depths, dtype=np.float64),
        dynamic_mask=np.asarray(dynamic_mask, dtype=np.bool_),
        sky_mask=np.asarray(sky_mask, dtype=np.bool_),
        c2w=c2w,
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
    )


def center_weights(num_frames: int) -> np.ndarray:
    indices = np.arange(num_frames)
    distance_to_edge = np.minimum(indices + 1, num_frames - indices)
    return distance_to_edge.astype(np.float64)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def weighted_rotation_mean(matrices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrices).mean(weights=weights).as_matrix()


def estimate_depth_scale(
    clip: ClipData,
    overlap_global: np.ndarray,
    global_start: int,
    global_depths: np.ndarray,
    global_dynamic: np.ndarray,
    global_sky: np.ndarray,
    max_samples_per_frame: int,
) -> Tuple[Optional[float], int]:
    frame_log_scales = []
    total_samples = 0
    for global_frame in overlap_global:
        global_idx = int(global_frame - global_start)
        local_idx = int(global_frame - clip.global_start)
        depth_global = global_depths[global_idx]
        depth_local = clip.depths[local_idx]
        valid = (
            np.isfinite(depth_global)
            & np.isfinite(depth_local)
            & (depth_global > 1e-6)
            & (depth_local > 1e-6)
            & ~global_dynamic[global_idx]
            & ~clip.dynamic_mask[local_idx]
            & ~global_sky[global_idx]
            & ~clip.sky_mask[local_idx]
        )
        ratios = depth_global[valid] / depth_local[valid]
        if ratios.size == 0:
            continue
        if ratios.size > max_samples_per_frame:
            stride = max(1, ratios.size // max_samples_per_frame)
            ratios = ratios[::stride][:max_samples_per_frame]
        log_ratios = np.log(ratios)
        log_ratios = log_ratios[np.isfinite(log_ratios)]
        if log_ratios.size:
            frame_log_scales.append(float(np.median(log_ratios)))
            total_samples += int(log_ratios.size)
    if not frame_log_scales:
        return None, 0
    return float(np.exp(np.median(frame_log_scales))), total_samples


def estimate_camera_scale(global_centers: np.ndarray, local_centers: np.ndarray) -> Tuple[Optional[float], int]:
    ratios = []
    for first in range(len(global_centers)):
        for second in range(first + 1, len(global_centers)):
            global_distance = np.linalg.norm(global_centers[second] - global_centers[first])
            local_distance = np.linalg.norm(local_centers[second] - local_centers[first])
            if global_distance > 1e-6 and local_distance > 1e-6:
                ratios.append(global_distance / local_distance)
    if not ratios:
        return None, 0
    return float(np.median(ratios)), len(ratios)


def aggregate_pose(
    observations: List[Tuple[float, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.asarray([item[0] for item in observations], dtype=np.float64)
    c2w_values = np.stack([item[1] for item in observations])
    intrinsics_values = np.stack([item[2] for item in observations])
    rotation = weighted_rotation_mean(c2w_values[:, :3, :3], weights)
    translation = np.average(c2w_values[:, :3, 3], axis=0, weights=weights)
    intrinsics = np.average(intrinsics_values, axis=0, weights=weights)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation
    return c2w, intrinsics


def align_clip(
    clip: ClipData,
    overlap_global: np.ndarray,
    full_start: int,
    pose_observations: List[List[Tuple[float, np.ndarray, np.ndarray]]],
    global_depths: np.ndarray,
    global_dynamic: np.ndarray,
    global_sky: np.ndarray,
    min_scale: float,
    max_scale: float,
    max_depth_samples_per_frame: int,
) -> Tuple[float, np.ndarray, np.ndarray, Dict]:
    global_poses = []
    local_poses = []
    pose_weights = []
    local_weights = center_weights(len(clip.c2w))
    for global_frame in overlap_global:
        global_idx = int(global_frame - full_start)
        local_idx = int(global_frame - clip.global_start)
        pose, _ = aggregate_pose(pose_observations[global_idx])
        global_poses.append(pose)
        local_poses.append(clip.c2w[local_idx])
        pose_weights.append(local_weights[local_idx])

    global_poses = np.stack(global_poses)
    local_poses = np.stack(local_poses)
    pose_weights = np.asarray(pose_weights, dtype=np.float64)

    rotation_candidates = (
        global_poses[:, :3, :3]
        @ np.swapaxes(local_poses[:, :3, :3], 1, 2)
    )
    world_rotation = weighted_rotation_mean(rotation_candidates, pose_weights)

    depth_scale, depth_samples = estimate_depth_scale(
        clip,
        overlap_global,
        full_start,
        global_depths,
        global_dynamic,
        global_sky,
        max_depth_samples_per_frame,
    )
    camera_scale, camera_pairs = estimate_camera_scale(
        global_poses[:, :3, 3],
        local_poses[:, :3, 3],
    )

    scale_source = "depth"
    scale = depth_scale
    if scale is None or not np.isfinite(scale) or not min_scale <= scale <= max_scale:
        scale_source = "camera"
        scale = camera_scale
    if scale is None or not np.isfinite(scale) or not min_scale <= scale <= max_scale:
        scale_source = "identity"
        scale = 1.0
    scale = float(np.clip(scale, min_scale, max_scale))

    transformed_centers = scale * (
        world_rotation @ local_poses[:, :3, 3, None]
    )[:, :, 0]
    translations = global_poses[:, :3, 3] - transformed_centers
    world_translation = np.average(translations, axis=0, weights=pose_weights)

    aligned_overlap_rotations = world_rotation[None] @ local_poses[:, :3, :3]
    rotation_errors = [
        rotation_angle_deg(global_poses[i, :3, :3].T @ aligned_overlap_rotations[i])
        for i in range(len(overlap_global))
    ]
    aligned_overlap_centers = scale * (
        world_rotation @ local_poses[:, :3, 3, None]
    )[:, :, 0] + world_translation
    translation_errors = np.linalg.norm(
        global_poses[:, :3, 3] - aligned_overlap_centers,
        axis=1,
    )

    diagnostics = {
        "overlap_global_frames": [int(value) for value in overlap_global],
        "scale": scale,
        "scale_source": scale_source,
        "depth_scale_candidate": depth_scale,
        "depth_scale_samples": depth_samples,
        "camera_scale_candidate": camera_scale,
        "camera_scale_pairs": camera_pairs,
        "rotation_error_mean_deg": float(np.mean(rotation_errors)),
        "rotation_error_max_deg": float(np.max(rotation_errors)),
        "translation_error_rms": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_error_max": float(np.max(translation_errors)),
    }
    return scale, world_rotation, world_translation, diagnostics


def transform_c2w(
    c2w: np.ndarray,
    scale: float,
    world_rotation: np.ndarray,
    world_translation: np.ndarray,
) -> np.ndarray:
    transformed = np.repeat(np.eye(4, dtype=np.float64)[None], len(c2w), axis=0)
    transformed[:, :3, :3] = world_rotation[None] @ c2w[:, :3, :3]
    transformed[:, :3, 3] = (
        scale * (world_rotation[None] @ c2w[:, :3, 3, None])[:, :, 0]
        + world_translation
    )
    return transformed


def prepare_output_folder(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output folder exists: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def save_lossless_video(path: Path, video: np.ndarray, fps: float) -> None:
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


def main(args) -> None:
    root = repo_root()
    manifest_path = resolve_path(args.manifest, root)
    input_result_root = resolve_path(args.input_result_root, root)
    output_folder = resolve_path(args.output_folder, root)
    manifest, clip_entries, full_start, full_end = load_manifest(manifest_path)
    full_frames = full_end - full_start

    clips = [
        load_clip(entry, input_result_root, args.resolution, args.num_frames)
        for entry in clip_entries
    ]
    first_shape = clips[0].video.shape[1:]
    for clip in clips:
        if clip.video.shape[1:] != first_shape:
            raise ValueError(
                f"Clip {clip.index} video shape {clip.video.shape[1:]} does not match {first_shape}"
            )
        if abs(clip.fps - clips[0].fps) > 0.1:
            raise ValueError(f"Clip {clip.index} FPS {clip.fps} does not match {clips[0].fps}")

    height, width, channels = first_shape
    global_video = np.zeros((full_frames, height, width, channels), dtype=np.uint8)
    global_depths = np.zeros((full_frames, height, width), dtype=np.float64)
    global_dynamic = np.zeros((full_frames, height, width), dtype=np.bool_)
    global_sky = np.zeros((full_frames, height, width), dtype=np.bool_)
    owner_weight = np.full(full_frames, -np.inf, dtype=np.float64)
    owner_clip = np.full(full_frames, -1, dtype=np.int64)
    pose_observations: List[List[Tuple[float, np.ndarray, np.ndarray]]] = [
        [] for _ in range(full_frames)
    ]
    clip_reports = []

    for sequence_index, clip in enumerate(clips):
        covered = np.asarray(
            [
                global_frame
                for global_frame in range(clip.global_start, clip.global_end + 1)
                if pose_observations[global_frame - full_start]
            ],
            dtype=np.int64,
        )
        if sequence_index == 0:
            scale = 1.0
            world_rotation = np.eye(3, dtype=np.float64)
            world_translation = np.zeros(3, dtype=np.float64)
            diagnostics = {
                "overlap_global_frames": [],
                "scale": scale,
                "scale_source": "anchor",
                "depth_scale_candidate": None,
                "depth_scale_samples": 0,
                "camera_scale_candidate": None,
                "camera_scale_pairs": 0,
                "rotation_error_mean_deg": 0.0,
                "rotation_error_max_deg": 0.0,
                "translation_error_rms": 0.0,
                "translation_error_max": 0.0,
            }
        else:
            if len(covered) < args.min_overlap_frames:
                raise ValueError(
                    f"Clip {clip.index} overlaps the assembled sequence by only {len(covered)} frames; "
                    f"need at least {args.min_overlap_frames} for Sim(3) alignment."
                )
            scale, world_rotation, world_translation, diagnostics = align_clip(
                clip,
                covered,
                full_start,
                pose_observations,
                global_depths,
                global_dynamic,
                global_sky,
                args.min_scale,
                args.max_scale,
                args.max_depth_samples_per_frame,
            )

        aligned_c2w = transform_c2w(
            clip.c2w,
            scale,
            world_rotation,
            world_translation,
        )
        aligned_depths = clip.depths * scale
        weights = center_weights(len(clip.video))

        for local_idx, global_frame in enumerate(
            range(clip.global_start, clip.global_end + 1)
        ):
            global_idx = global_frame - full_start
            weight = float(weights[local_idx])
            pose_observations[global_idx].append(
                (weight, aligned_c2w[local_idx], clip.intrinsics[local_idx])
            )
            if weight > owner_weight[global_idx]:
                owner_weight[global_idx] = weight
                owner_clip[global_idx] = clip.index
                global_video[global_idx] = clip.video[local_idx]
                global_depths[global_idx] = aligned_depths[local_idx]
                global_dynamic[global_idx] = clip.dynamic_mask[local_idx]
                global_sky[global_idx] = clip.sky_mask[local_idx]

        report = {
            "clip_index": clip.index,
            "input_folder": str(clip.folder),
            "global_start_frame": clip.global_start,
            "global_end_frame": clip.global_end,
            "similarity_scale": scale,
            "world_rotation": world_rotation.tolist(),
            "world_translation": world_translation.tolist(),
            **diagnostics,
        }
        clip_reports.append(report)
        print(
            f"[{clip.index:03d}] {clip.global_start}..{clip.global_end}: "
            f"overlap={len(covered)}, scale={scale:.6f} ({diagnostics['scale_source']}), "
            f"rot_mean={diagnostics['rotation_error_mean_deg']:.4f} deg, "
            f"trans_rms={diagnostics['translation_error_rms']:.6f}"
        )

    missing_frames = [
        int(full_start + index)
        for index, observations in enumerate(pose_observations)
        if not observations
    ]
    if missing_frames:
        raise ValueError(f"Manifest clips do not cover global frames: {missing_frames}")

    global_c2w = np.empty((full_frames, 4, 4), dtype=np.float64)
    global_intrinsics = np.empty((full_frames, clips[0].intrinsics.shape[1]), dtype=np.float64)
    observation_counts = []
    for frame_idx, observations in enumerate(pose_observations):
        global_c2w[frame_idx], global_intrinsics[frame_idx] = aggregate_pose(observations)
        observation_counts.append(len(observations))

    prepare_output_folder(output_folder, overwrite=args.overwrite)
    save_lossless_video(output_folder / "video.mp4", global_video, clips[0].fps)
    save_depths(str(output_folder / "depths"), global_depths, dtype=np.float16)
    save_masks(str(output_folder / "dynamic_mask"), global_dynamic)
    save_masks(str(output_folder / "sky_mask"), global_sky)
    save_cameras(
        str(output_folder / "cameras.npz"),
        global_c2w.astype(np.float32),
        global_intrinsics.astype(np.float32),
    )
    save_clips(str(output_folder / "clips.json"), {"src": (0, full_frames)})

    report = {
        "source": "stitched_split_reconstructions",
        "manifest": str(manifest_path),
        "input_result_root": str(input_result_root),
        "output_folder": str(output_folder),
        "full_start_frame": full_start,
        "full_end_exclusive": full_end,
        "full_frames": full_frames,
        "num_clips": len(clips),
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "content_merge": "center_owner_hard_cut",
        "camera_merge": "weighted_rotation_translation_average",
        "owner_clip_by_frame": owner_clip.tolist(),
        "camera_observation_counts": observation_counts,
        "clips": clip_reports,
    }
    report_path = output_folder / "stitch_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Stitched {len(clips)} clips into {full_frames} frames: {output_folder}")
    print(f"Stitch report: {report_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Align independently reconstructed Vista4D clips with overlap Sim(3), "
            "then merge them into one full-sequence reconstruction."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input_result_root", default="./results/single")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--resolution", default="384p")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--min_overlap_frames", type=int, default=2)
    parser.add_argument("--min_scale", type=float, default=0.1)
    parser.add_argument("--max_scale", type=float, default=10.0)
    parser.add_argument("--max_depth_samples_per_frame", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true", default=False)
    main(parser.parse_args())
