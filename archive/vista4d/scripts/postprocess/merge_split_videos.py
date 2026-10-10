from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np

from utils.split_manifest import (
    clip_num_frames,
    clip_pad_right,
    clip_valid_num_frames,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def read_manifest(path: Path) -> Tuple[List[Dict], Optional[float], Optional[int]]:
    if path.suffix == ".json":
        with path.open("r") as f:
            data = json.load(f)
        clips = data["clips"]
        return clips, data.get("fps"), data.get("total_frames")

    with path.open("r", newline="") as f:
        clips = list(csv.DictReader(f))
    for clip in clips:
        for key in (
            "clip_index",
            "start_frame",
            "end_frame",
            "num_frames",
            "valid_num_frames",
            "pad_right",
            "padded_end_frame",
        ):
            if key in clip and clip[key] not in (None, ""):
                clip[key] = int(clip[key])
    return clips, None, None


def prepare_clip_video(
    video: np.ndarray,
    clip: Dict,
    strict_num_frames: bool,
    path: Path,
) -> np.ndarray:
    encoded_frames = clip_num_frames(clip)
    valid_frames = clip_valid_num_frames(clip)
    if strict_num_frames and video.shape[0] != encoded_frames:
        raise ValueError(
            f"{path} has {video.shape[0]} frames, expected encoded "
            f"num_frames={encoded_frames}"
        )
    if video.shape[0] < valid_frames:
        raise ValueError(
            f"{path} has {video.shape[0]} frames, fewer than "
            f"valid_num_frames={valid_frames}"
        )
    return video[:valid_frames]


def derive_example_name(split_video_path: str, resolution: str, num_frames: int) -> str:
    stem = Path(split_video_path).stem
    suffix_384 = f"_384p{num_frames}"
    suffix_720 = f"_720p{num_frames}"
    if stem.endswith(suffix_384):
        return stem[: -len(suffix_384)] + f"_{resolution}{num_frames}"
    if stem.endswith(suffix_720):
        return stem[: -len(suffix_720)] + f"_{resolution}{num_frames}"
    return stem


def find_clip_video(
    result_root: Path,
    example: str,
    inference_folder: str,
    video_name: str,
    seed: Optional[str],
) -> Path:
    folder = result_root / example / inference_folder
    if video_name != "auto":
        return folder / video_name

    if seed:
        return folder / f"video_seed={seed}.mp4"

    candidates = sorted(folder.glob("video_seed=*.mp4"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No video_seed=*.mp4 found in {folder}")
    raise ValueError(f"Multiple video_seed=*.mp4 files found in {folder}; pass --seed or --video_name.")


def load_video(path: Path) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0), fps


def save_video(path: Path, video: np.ndarray, fps: float, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, quality=quality, macro_block_size=1)
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _limit_flow(flow: np.ndarray, max_pixels: float) -> np.ndarray:
    if max_pixels <= 0:
        return flow
    magnitude = np.linalg.norm(flow, axis=2, keepdims=True)
    scale = np.minimum(1.0, max_pixels / np.maximum(magnitude, 1e-6))
    return flow * scale


def _remap_with_flow(frame: np.ndarray, flow: np.ndarray, amount: float) -> np.ndarray:
    height, width = frame.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0] * amount
    map_y = grid_y + flow[..., 1] * amount
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _dense_flow(source: np.ndarray, target: np.ndarray, max_pixels: float) -> np.ndarray:
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        source_gray,
        target_gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    return _limit_flow(flow, max_pixels)


def _flow_blend_pair(
    previous: np.ndarray,
    current: np.ndarray,
    weight: float,
    max_pixels: float,
    consistency_sigma: float,
) -> np.ndarray:
    forward = _dense_flow(previous, current, max_pixels)
    backward = _dense_flow(current, previous, max_pixels)

    current_at_previous = _remap_with_flow(current, forward, 1.0)
    backward_at_previous = _remap_with_flow(backward, forward, 1.0)
    consistency_error = np.linalg.norm(forward + backward_at_previous, axis=2)
    consistency = np.exp(-0.5 * (consistency_error / max(consistency_sigma, 1e-6)) ** 2)

    photo_error = np.mean(
        np.abs(previous.astype(np.float32) - current_at_previous.astype(np.float32)),
        axis=2,
    )
    photometric = np.exp(-0.5 * (photo_error / 48.0) ** 2)
    confidence = cv2.GaussianBlur(consistency * photometric, (0, 0), sigmaX=1.5)
    confidence = np.clip(confidence, 0.0, 1.0)[..., None].astype(np.float32)

    # Move both frames toward an intermediate geometry before blending them.
    previous_moved = _remap_with_flow(previous, backward, weight).astype(np.float32)
    current_moved = _remap_with_flow(current, forward, 1.0 - weight).astype(np.float32)
    aligned = previous_moved * (1.0 - weight) + current_moved * weight

    # Where flow is unreliable, make a short handoff around the overlap midpoint.
    # This avoids the prolonged double image produced by a full linear dissolve.
    fallback_weight = _smoothstep((weight - 0.40) / 0.20)
    fallback = (
        previous.astype(np.float32) * (1.0 - fallback_weight)
        + current.astype(np.float32) * fallback_weight
    )
    blended = aligned * confidence + fallback * (1.0 - confidence)
    return np.rint(blended).clip(0, 255).astype(np.uint8)


def flow_blend_overlap(
    previous: np.ndarray,
    current: np.ndarray,
    max_pixels: float,
    consistency_sigma: float,
) -> np.ndarray:
    n = previous.shape[0]
    result = np.empty_like(previous)
    for index in range(n):
        weight = _smoothstep((index + 1) / (n + 1))
        result[index] = _flow_blend_pair(
            previous[index],
            current[index],
            weight=weight,
            max_pixels=max_pixels,
            consistency_sigma=consistency_sigma,
        )
    return result


def blend_overlap(
    previous: np.ndarray,
    current: np.ndarray,
    mode: str,
    flow_max_pixels: float,
    flow_consistency_sigma: float,
) -> np.ndarray:
    if mode == "center_cut":
        previous_count = (previous.shape[0] + 1) // 2
        return np.concatenate(
            [previous[:previous_count], current[previous_count:]],
            axis=0,
        )
    if mode == "trim":
        return previous
    if mode == "replace":
        return current
    if mode == "average":
        return np.rint(previous.astype(np.float32) * 0.5 + current.astype(np.float32) * 0.5).astype(np.uint8)
    if mode == "blend":
        return flow_blend_overlap(
            previous,
            current,
            max_pixels=flow_max_pixels,
            consistency_sigma=flow_consistency_sigma,
        )
    raise ValueError(f"Unknown merge mode: {mode}")


def merge_videos(
    clips: List[Dict],
    clip_videos: List[np.ndarray],
    mode: str,
    flow_max_pixels: float,
    flow_consistency_sigma: float,
) -> Tuple[np.ndarray, List[Dict]]:
    order = np.argsort([int(clip["start_frame"]) for clip in clips])
    clips = [clips[i] for i in order]
    clip_videos = [clip_videos[i] for i in order]

    output_start = int(clips[0]["start_frame"])
    output = clip_videos[0].copy()
    current_end = output_start + output.shape[0] - 1
    merge_log = []

    for clip, video in zip(clips[1:], clip_videos[1:]):
        start = int(clip["start_frame"])
        end = start + video.shape[0] - 1

        if start > current_end + 1:
            raise ValueError(f"Gap between clips: current_end={current_end}, next_start={start}")

        if start <= current_end:
            overlap_len = min(current_end - start + 1, video.shape[0])
            previous_overlap_frames = overlap_len
            current_overlap_frames = 0
            if mode == "center_cut":
                previous_overlap_frames = (overlap_len + 1) // 2
                current_overlap_frames = overlap_len - previous_overlap_frames
            elif mode == "replace":
                previous_overlap_frames = 0
                current_overlap_frames = overlap_len
            elif mode in ("average", "blend"):
                current_overlap_frames = overlap_len
            output_offset = start - output_start
            previous_overlap = output[output_offset : output_offset + overlap_len]
            current_overlap = video[:overlap_len]
            output[output_offset : output_offset + overlap_len] = blend_overlap(
                previous_overlap,
                current_overlap,
                mode,
                flow_max_pixels=flow_max_pixels,
                flow_consistency_sigma=flow_consistency_sigma,
            )
            output = np.concatenate([output, video[overlap_len:]], axis=0)
        else:
            overlap_len = 0
            previous_overlap_frames = 0
            current_overlap_frames = 0
            output = np.concatenate([output, video], axis=0)

        merge_log.append(
            {
                "clip_index": int(clip["clip_index"]),
                "start_frame": start,
                "end_frame": int(clip["end_frame"]),
                "valid_frames": int(video.shape[0]),
                "pad_right_trimmed": clip_pad_right(clip),
                "overlap_frames": int(overlap_len),
                "previous_overlap_frames": int(previous_overlap_frames),
                "current_overlap_frames": int(current_overlap_frames),
                "output_frames_after_clip": int(output.shape[0]),
            }
        )
        current_end = max(current_end, end)

    return output, merge_log


def main(args) -> None:
    root = repo_root()
    manifest_path = resolve_path(args.manifest, root)
    result_root = resolve_path(args.result_root, root)
    output_path = resolve_path(args.output, root)

    clips, manifest_fps, _ = read_manifest(manifest_path)
    if not clips:
        raise ValueError(f"No clips found in manifest: {manifest_path}")

    clips = sorted(clips, key=lambda row: int(row["clip_index"]))
    videos = []
    video_paths = []
    fps_values = []
    expected_hw = None

    for clip in clips:
        example = derive_example_name(clip["output_path"], args.resolution, args.num_frames)
        clip_video_path = find_clip_video(
            result_root=result_root,
            example=example,
            inference_folder=args.inference_folder,
            video_name=args.video_name,
            seed=args.seed,
        )
        if not clip_video_path.is_file():
            raise FileNotFoundError(f"Generated clip video not found: {clip_video_path}")

        video, fps = load_video(clip_video_path)
        video = prepare_clip_video(
            video,
            clip,
            strict_num_frames=args.strict_num_frames,
            path=clip_video_path,
        )
        if expected_hw is None:
            expected_hw = video.shape[1:3]
        elif video.shape[1:3] != expected_hw:
            raise ValueError(f"{clip_video_path} has shape {video.shape[1:3]}, expected {expected_hw}")

        videos.append(video)
        video_paths.append(clip_video_path)
        fps_values.append(fps)

    fps = args.fps if args.fps is not None else manifest_fps or fps_values[0]
    merged, merge_log = merge_videos(
        clips,
        videos,
        args.merge_mode,
        flow_max_pixels=args.flow_max_pixels,
        flow_consistency_sigma=args.flow_consistency_sigma,
    )

    first_start = min(int(clip["start_frame"]) for clip in clips)
    last_end = max(int(clip["end_frame"]) for clip in clips)
    expected_total = last_end - first_start + 1
    if args.strict_total_frames and merged.shape[0] != expected_total:
        raise ValueError(f"Merged video has {merged.shape[0]} frames, expected {expected_total}")

    save_video(output_path, merged, fps=fps, quality=args.quality)

    report = {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "merge_mode": args.merge_mode,
        "flow_max_pixels": args.flow_max_pixels,
        "flow_consistency_sigma": args.flow_consistency_sigma,
        "fps": float(fps),
        "frames": int(merged.shape[0]),
        "height": int(merged.shape[1]),
        "width": int(merged.shape[2]),
        "clip_videos": [str(path) for path in video_paths],
        "merge_log": merge_log,
    }
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Merged {len(videos)} clips")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Frames: {merged.shape[0]}, fps: {fps:.6f}, size: {merged.shape[2]}x{merged.shape[1]}")
    for item in merge_log:
        print(
            f"clip {item['clip_index']}: start={item['start_frame']} end={item['end_frame']} "
            f"valid={item['valid_frames']} pad_trimmed={item['pad_right_trimmed']} "
            f"overlap={item['overlap_frames']} "
            f"previous/current={item['previous_overlap_frames']}/{item['current_overlap_frames']} "
            f"output_frames={item['output_frames_after_clip']}"
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Merge Vista4D split outputs back to one video.")
    parser.add_argument("--manifest", required=True, help="Split manifest CSV or JSON.")
    parser.add_argument("--result_root", default="./results/single")
    parser.add_argument("--resolution", default="384p")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--inference_folder", default="vista4d_384p_smooth")
    parser.add_argument("--video_name", default="auto", help="Generated clip video name, or auto for video_seed=*.mp4.")
    parser.add_argument("--seed", default=None, help="Seed used when --video_name=auto, e.g. 10027.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--merge_mode",
        default="center_cut",
        choices=["center_cut", "blend", "trim", "replace", "average"],
    )
    parser.add_argument(
        "--flow_max_pixels",
        type=float,
        default=32.0,
        help="Maximum optical-flow displacement used by blend mode; <=0 disables clipping.",
    )
    parser.add_argument(
        "--flow_consistency_sigma",
        type=float,
        default=2.5,
        help="Forward/backward flow consistency tolerance in pixels for blend mode.",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--quality", type=int, default=9)
    parser.add_argument("--strict_num_frames", action="store_true", default=True)
    parser.add_argument("--no_strict_num_frames", dest="strict_num_frames", action="store_false")
    parser.add_argument("--strict_total_frames", action="store_true", default=True)
    parser.add_argument("--no_strict_total_frames", dest="strict_total_frames", action="store_false")
    main(parser.parse_args())
