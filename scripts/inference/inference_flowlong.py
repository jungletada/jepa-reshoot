from argparse import ArgumentParser
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from time import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import yaml

from diffsynth.pipelines.flowlong import (
    FlowLongSamplingConfig,
    build_geometry_from_manifest,
)
from scripts.inference.inference import get_pipeline
from utils.media import load_cameras, load_masks, load_video, save_video
from utils.resolution import (
    validate_manifest_resolution,
    validate_resolution_dimensions,
)
from utils.split_manifest import (
    clip_num_frames,
    clip_pad_right,
    clip_valid_num_frames,
)
from utils.vram_presets import add_vram_args


DEFAULT_NEGATIVE_PROMPT = (
    "gaudy, overexposed, static, blurred details, subtitles, style, artwork, "
    "painting, still, overall gray, worst quality, low quality, JPEG compression, "
    "ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, malformed limbs, fused fingers, still image, cluttered "
    "background, three legs, many people in background, walking backwards"
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_metadata(root: Path) -> Dict:
    def run(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
    }


def exact_edge_repeat(value: np.ndarray, valid_frames: int) -> bool:
    if valid_frames == value.shape[0]:
        return True
    expected = np.repeat(
        value[valid_frames - 1 : valid_frames],
        value.shape[0] - valid_frames,
        axis=0,
    )
    actual = value[valid_frames:]
    if np.issubdtype(value.dtype, np.floating):
        return bool(np.array_equal(actual, expected, equal_nan=True))
    return bool(np.array_equal(actual, expected))


def load_required_masks(folder: Path, name: str, expected_frames: int) -> np.ndarray:
    path = folder / name
    if not path.is_dir():
        raise FileNotFoundError(path)
    masks = load_masks(str(path), desc=f"Loading {folder.parent.name}/{name}")
    if masks.shape[0] != expected_frames:
        raise ValueError(
            f"{path} has {masks.shape[0]} frames, expected {expected_frames}"
        )
    return masks


def load_flowlong_inputs(
    manifest: Dict,
    result_root: Path,
    render_folder: str,
    resolution: str,
    num_frames: int,
    height: int,
    width: int,
) -> Tuple[Dict, float, List[str], str]:
    clips = sorted(manifest["clips"], key=lambda clip: int(clip["clip_index"]))
    source_videos = []
    point_cloud_videos = []
    source_alpha_masks = []
    source_motion_masks = []
    point_cloud_alpha_masks = []
    point_cloud_motion_masks = []
    target_cameras = []
    target_intrinsics = []
    fps_values = []
    condition_folders = []
    shared_render_folder = None

    for clip in clips:
        clip_index = int(clip["clip_index"])
        expected_frames = clip_num_frames(clip)
        valid_frames = clip_valid_num_frames(clip)
        pad_right = clip_pad_right(clip)
        if expected_frames != num_frames:
            raise ValueError(
                f"Clip {clip_index} has {expected_frames} frames, expected {num_frames}"
            )
        example = derive_example_name(clip["output_path"], resolution, num_frames)
        folder = result_root / example / render_folder
        if not folder.is_dir():
            raise FileNotFoundError(folder)

        metadata_path = folder / "full_shared_static_slice.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "source": "full_shared_static_render_slice",
            "clip_index": clip_index,
            "global_start_frame": int(clip["start_frame"]),
            "global_end_frame": int(clip["end_frame"]),
            "frames": expected_frames,
            "valid_frames": valid_frames,
            "pad_right": pad_right,
            "padding_mode": "edge",
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"{metadata_path}: expected {name}={expected!r}, "
                    f"got {metadata.get(name)!r}"
                )
        folder_provenance = metadata.get("full_render_folder")
        if not folder_provenance:
            raise ValueError(f"Missing full_render_folder in {metadata_path}")
        if shared_render_folder is None:
            shared_render_folder = folder_provenance
        elif folder_provenance != shared_render_folder:
            raise ValueError(
                f"Clip {clip_index} comes from {folder_provenance}, expected "
                f"{shared_render_folder}"
            )

        video_src, fps_src = load_video(
            str(folder / "video_src.mp4"),
            desc=f"Loading {example}/video_src",
        )
        video_pc, fps_pc = load_video(
            str(folder / "video_pc.mp4"),
            desc=f"Loading {example}/video_pc",
        )
        for name, value in (("video_src", video_src), ("video_pc", video_pc)):
            if value.shape[0] != expected_frames:
                raise ValueError(
                    f"{folder}/{name}.mp4 has {value.shape[0]} frames, "
                    f"expected {expected_frames}"
                )
            if tuple(value.shape[1:3]) != (height, width):
                raise ValueError(
                    f"{folder}/{name}.mp4 has spatial shape "
                    f"{value.shape[2]}x{value.shape[1]}, expected {width}x{height} "
                    f"for resolution={resolution}"
                )
            if pad_right and not exact_edge_repeat(value, valid_frames):
                raise ValueError(
                    f"{folder}/{name}.mp4 does not edge-repeat its {pad_right} "
                    "padding frames"
                )
        fps_values.extend((float(fps_src), float(fps_pc)))

        source_alpha = load_required_masks(folder, "alpha_mask_src", expected_frames)
        source_motion = load_required_masks(folder, "dynamic_mask_src", expected_frames)
        point_cloud_alpha = load_required_masks(
            folder, "alpha_mask_pc", expected_frames
        )
        point_cloud_motion = load_required_masks(
            folder, "dynamic_mask_pc", expected_frames
        )
        for name, value in (
            ("alpha_mask_src", source_alpha),
            ("dynamic_mask_src", source_motion),
            ("alpha_mask_pc", point_cloud_alpha),
            ("dynamic_mask_pc", point_cloud_motion),
        ):
            if tuple(value.shape[1:3]) != (height, width):
                raise ValueError(
                    f"{folder}/{name} has spatial shape "
                    f"{value.shape[2]}x{value.shape[1]}, expected "
                    f"{width}x{height} for resolution={resolution}"
                )
            if pad_right and not exact_edge_repeat(value, valid_frames):
                raise ValueError(
                    f"{folder}/{name} does not edge-repeat its {pad_right} padding frames"
                )

        camera_path = folder / "cameras_tgt.npz"
        if not camera_path.is_file():
            raise FileNotFoundError(camera_path)
        cam_c2w, intrinsics = load_cameras(str(camera_path))
        if cam_c2w.shape[0] != expected_frames or intrinsics.shape[0] != expected_frames:
            raise ValueError(f"{camera_path} does not contain {expected_frames} frames")
        if pad_right and (
            not exact_edge_repeat(cam_c2w, valid_frames)
            or not exact_edge_repeat(intrinsics, valid_frames)
        ):
            raise ValueError(
                f"{camera_path} does not edge-repeat its {pad_right} padding frames"
            )

        source_videos.append([Image.fromarray(frame) for frame in video_src])
        point_cloud_videos.append([Image.fromarray(frame) for frame in video_pc])
        source_alpha_masks.append(source_alpha)
        source_motion_masks.append(source_motion)
        point_cloud_alpha_masks.append(point_cloud_alpha)
        point_cloud_motion_masks.append(point_cloud_motion)
        target_cameras.append(cam_c2w)
        target_intrinsics.append(intrinsics)
        condition_folders.append(str(folder))

    if max(fps_values) - min(fps_values) > 1e-3:
        raise ValueError(f"Condition video FPS values do not match: {fps_values}")
    return {
        "source_video": source_videos,
        "point_cloud_video": point_cloud_videos,
        "source_alpha_mask": np.stack(source_alpha_masks),
        "source_motion_mask": np.stack(source_motion_masks),
        "point_cloud_alpha_mask": np.stack(point_cloud_alpha_masks),
        "point_cloud_motion_mask": np.stack(point_cloud_motion_masks),
        "target_cam_c2w": np.stack(target_cameras),
        "target_intrinsics": np.stack(target_intrinsics),
    }, fps_values[0], condition_folders, str(shared_render_folder)


@torch.no_grad()
def main(args) -> None:
    root = repo_root()
    validate_resolution_dimensions(
        args.resolution,
        height=args.height,
        width=args.width,
    )
    manifest_path = resolve_path(args.manifest, root)
    result_root = resolve_path(args.result_root, root)
    output_folder = resolve_path(args.output_folder, root)
    from utils.vista4d_checkpoint import resolve_checkpoint, checkpoint_sha256 as hash_checkpoint, checkpoint_size
    vista4d_checkpoint = resolve_checkpoint(resolve_path(args.vista4d_checkpoint, root)).path
    vista4d_config_path = resolve_path(args.vista4d_config_path, root)
    if manifest_path.suffix.lower() != ".json":
        raise ValueError("FlowLong requires a JSON manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_resolution(
        manifest,
        resolution=args.resolution,
        height=args.height,
        width=args.width,
    )
    geometry = build_geometry_from_manifest(manifest, temporal_factor=4)
    inputs, fps, condition_folders, shared_render_folder = load_flowlong_inputs(
        manifest,
        result_root=result_root,
        render_folder=args.render_folder,
        resolution=args.resolution,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    inputs.update(
        {
            "prompt": [args.prompt] * geometry.num_windows,
            "negative_prompt": [args.negative_prompt] * geometry.num_windows,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
        }
    )

    with vista4d_config_path.open("r", encoding="utf-8") as file:
        vista4d_config = yaml.safe_load(file)
    inputs.update(
        {
            "source_noise_level": vista4d_config["dit"]["augmentation"][
                "source_noise_level"
            ],
            "point_cloud_noise_level": vista4d_config["dit"]["augmentation"][
                "point_cloud_noise_level"
            ],
            "image_noise_level": vista4d_config["dit"]["augmentation"][
                "image_noise_level"
            ],
        }
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    for seed in args.seed:
        video_path = output_folder / f"video_seed={seed}.mp4"
        report_path = output_folder / f"flowlong_report_seed={seed}.json"
        if not args.overwrite and (video_path.exists() or report_path.exists()):
            raise FileExistsError(
                f"Output exists for seed {seed}. Pass --overwrite to replace it: "
                f"{video_path}"
            )

    args.vista4d_checkpoint = str(vista4d_checkpoint)
    args.vista4d_config_path = str(vista4d_config_path)
    checkpoint_sha256 = args.vista4d_checkpoint_sha256
    if checkpoint_sha256 is None:
        checkpoint_sha256 = hash_checkpoint(vista4d_checkpoint)
    checkpoint_sha256 = checkpoint_sha256.lower()
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        raise ValueError("--vista4d_checkpoint_sha256 must be a 64-digit hex digest")

    pipe = get_pipeline(args, vista4d_config)
    sampling_config = FlowLongSamplingConfig(
        stochastic_threshold=args.flowlong_stochastic_threshold,
        stochastic_enabled=not args.flowlong_disable_stochastic,
        microbatch_size=args.flowlong_microbatch_size,
    )
    manifest_sha256 = sha256_file(manifest_path)
    repository = git_metadata(root)
    checkpoint_info = {
        "path": str(vista4d_checkpoint),
        "size_bytes": checkpoint_size(vista4d_checkpoint),
        "sha256": checkpoint_sha256,
    }

    for seed in args.seed:
        started = time()
        video, pipeline_report = pipe.generate_flowlong(
            flowlong_geometry=geometry,
            flowlong_config=sampling_config,
            base_seed=seed,
            **inputs,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            tiled=args.tile_vae,
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
        )
        video_array = np.stack([np.asarray(frame) for frame in video], axis=0)
        video_path = output_folder / f"video_seed={seed}.mp4"
        save_video(
            str(video_path), video_array, fps=fps, quality=9,
            imageio_params={"input_params": ["-r", format(float(fps), ".12g")]},
        )
        video_sha256 = sha256_file(video_path)

        report = {
            "schema_version": 2,
            "paper": "FlowLong arXiv:2605.20910v1",
            "implementation": "vista4d-flowlong-v1",
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha256,
            },
            "geometry": asdict(geometry),
            "conditions": {
                "result_root": str(result_root),
                "render_folder": args.render_folder,
                "window_folders": condition_folders,
                "full_render_folder": shared_render_folder,
            },
            "model": {
                "model_id_with_origin_paths": args.model_id_with_origin_paths,
                "local_model_folder": args.local_model_folder,
                "vista4d_checkpoint": checkpoint_info,
                "vista4d_config_path": str(vista4d_config_path),
            },
            "prompt": args.prompt,
            "prompt_sha256": sha256_text(args.prompt),
            "negative_prompt_sha256": sha256_text(args.negative_prompt),
            "experiment": {
                "seed": int(seed),
                "num_inference_steps": int(args.num_inference_steps),
                "sigma_shift": float(args.sigma_shift),
                "cfg_scale": float(args.cfg_scale),
                "cfg_merge": False,
                "use_usp": False,
                "stochastic_enabled": not args.flowlong_disable_stochastic,
                "stochastic_threshold": float(
                    args.flowlong_stochastic_threshold
                ),
                "microbatch_size": int(args.flowlong_microbatch_size),
            },
            "repository": repository,
            "resolution": args.resolution,
            "fps": fps,
            "output_video": str(video_path),
            "output_video_sha256": video_sha256,
            "output_frames": int(video_array.shape[0]),
            "height": int(video_array.shape[1]),
            "width": int(video_array.shape[2]),
            "wall_seconds": time() - started,
            "pipeline": pipeline_report,
        }
        report_path = output_folder / f"flowlong_report_seed={seed}.json"
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
        print(f"FlowLong video: {video_path}")
        print(f"FlowLong report: {report_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run joint-timestep FlowLong sampling for Vista4D windows."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result_root", default="./results/flowlong_single")
    parser.add_argument("--render_folder", default="render_384p_smooth")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--resolution", default="384p")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)

    parser.add_argument("--model_id_with_origin_paths", required=True)
    parser.add_argument("--tokenizer_id_with_origin_path", required=True)
    parser.add_argument("--local_model_folder", required=True)
    parser.add_argument("--vista4d_checkpoint", required=True)
    parser.add_argument("--vista4d_checkpoint_sha256", default=None)
    parser.add_argument("--vista4d_config_path", required=True)
    add_vram_args(parser)

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, nargs="+", default=[10027])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--flowlong_stochastic_threshold", type=float, default=0.6)
    parser.add_argument("--flowlong_disable_stochastic", action="store_true")
    parser.add_argument("--flowlong_microbatch_size", type=int, default=1)
    parser.add_argument("--tile_vae", action="store_true")
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(use_usp=False, cfg_merge=False)
    main(parser.parse_args())
