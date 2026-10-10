from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from scripts.inference.inference import get_inputs, get_pipeline
from scripts.inference.inference_flowlong import (
    DEFAULT_NEGATIVE_PROMPT,
    derive_example_name,
    git_metadata,
    repo_root,
    resolve_path,
    sha256_file,
    sha256_text,
)
from scripts.postprocess.merge_split_videos import (
    load_video,
    merge_videos,
    prepare_clip_video,
    save_video,
)
from utils.resolution import (
    validate_manifest_resolution,
    validate_resolution_dimensions,
)
from utils.vram_presets import add_vram_args
from utils.vista4d_checkpoint import resolve_checkpoint, checkpoint_sha256 as hash_checkpoint, checkpoint_size
from utils.split_manifest import (
    clip_num_frames,
    clip_pad_right,
    clip_valid_num_frames,
    validate_flowlong_window_manifest,
)


REQUIRED_CONDITION_FILES = ("video_src.mp4", "video_pc.mp4", "cameras_tgt.npz")
REQUIRED_CONDITION_FOLDERS = (
    "alpha_mask_src",
    "dynamic_mask_src",
    "alpha_mask_pc",
    "dynamic_mask_pc",
)


def canonical_sha256(value: Dict) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_text(payload)


def sha256_tree(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot hash empty condition tree: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_digest(value: str, name: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-digit hex digest")
    return value


def load_baseline_contract(
    args,
    manifest_path: Path,
    manifest: Dict,
    clips: List[Dict],
    condition_root: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> Tuple[Dict, List[Dict], str]:
    condition_entries = []
    shared_render_folder = None
    for clip in clips:
        clip_index = int(clip["clip_index"])
        example = derive_example_name(
            clip["output_path"], args.resolution, args.num_frames
        )
        folder = condition_root / example / args.render_folder
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for filename in REQUIRED_CONDITION_FILES:
            path = folder / filename
            if not path.is_file():
                raise FileNotFoundError(path)
        for dirname in REQUIRED_CONDITION_FOLDERS:
            path = folder / dirname
            if not path.is_dir():
                raise FileNotFoundError(path)

        metadata_path = folder / "full_shared_static_slice.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "source": "full_shared_static_render_slice",
            "clip_index": clip_index,
            "global_start_frame": int(clip["start_frame"]),
            "global_end_frame": int(clip["end_frame"]),
            "frames": clip_num_frames(clip),
            "valid_frames": clip_valid_num_frames(clip),
            "pad_right": clip_pad_right(clip),
            "padding_mode": "edge",
        }
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"{metadata_path}: expected {key}={expected_value!r}, "
                    f"got {metadata.get(key)!r}"
                )
        full_render_folder = metadata.get("full_render_folder")
        if not full_render_folder:
            raise ValueError(f"Missing full_render_folder in {metadata_path}")
        if shared_render_folder is None:
            shared_render_folder = str(full_render_folder)
        elif str(full_render_folder) != shared_render_folder:
            raise ValueError(
                f"Clip {clip_index} uses full render {full_render_folder}, expected "
                f"{shared_render_folder}"
            )
        condition_entries.append(
            {
                "clip_index": clip_index,
                "example": example,
                "folder": str(folder.resolve()),
                "slice_metadata": str(metadata_path.resolve()),
                "input_sha256": {
                    name: sha256_tree(folder / name)
                    for name in (
                        *REQUIRED_CONDITION_FILES,
                        *REQUIRED_CONDITION_FOLDERS,
                    )
                },
            }
        )

    contract = {
        "schema_version": 1,
        "kind": "independent_center_cut_baseline",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_start_frame": int(manifest.get("start_frame", clips[0]["start_frame"])),
        "manifest_end_exclusive": int(manifest["end_exclusive"]),
        "window_frames": int(args.num_frames),
        "pixel_overlap": 5,
        "pixel_stride": 44,
        "merge_mode": "center_cut",
        "seed": int(args.seed),
        "window_seeds": [int(args.seed)] * len(clips),
        "window_seed_policy": "same_seed_for_each_independent_window",
        "num_inference_steps": int(args.num_inference_steps),
        "sigma_shift": float(args.sigma_shift),
        "cfg_scale": float(args.cfg_scale),
        "cfg_merge": False,
        "use_usp": False,
        "prompt_sha256": sha256_text(args.prompt),
        "negative_prompt_sha256": sha256_text(args.negative_prompt),
        "vista4d_checkpoint_sha256": checkpoint_sha256,
        "vista4d_checkpoint_size_bytes": checkpoint_size(checkpoint_path),
        "full_render_folder": shared_render_folder,
        "condition_input_sha256": [
            entry["input_sha256"] for entry in condition_entries
        ],
    }
    return contract, condition_entries, shared_render_folder


def completed_clip(
    video_path: Path,
    report_path: Path,
    contract_sha256: str,
) -> Dict | None:
    if not video_path.is_file() or not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("contract_sha256") != contract_sha256:
        return None
    if report.get("output_video_sha256") != sha256_file(video_path):
        return None
    return report


def handoff_frames(merge_log: List[Dict]) -> List[int]:
    return [
        int(item["start_frame"]) + int(item["previous_overlap_frames"])
        for item in merge_log
        if int(item["overlap_frames"]) > 0
    ]


def import_existing_clip(
    *,
    args,
    clip: Dict,
    condition: Dict,
    existing_root: Path,
    report_path: Path,
    contract_sha256: str,
) -> Dict:
    clip_index = int(clip["clip_index"])
    example = condition["example"]
    existing_condition = existing_root / example / args.render_folder
    expected_condition = Path(condition["folder"])
    compared_assets = [
        *REQUIRED_CONDITION_FILES,
        *REQUIRED_CONDITION_FOLDERS,
    ]
    condition_hashes = {}
    for name in compared_assets:
        existing_path = existing_condition / name
        expected_path = expected_condition / name
        if not existing_path.exists():
            raise FileNotFoundError(existing_path)
        existing_sha256 = sha256_tree(existing_path)
        expected_sha256 = sha256_tree(expected_path)
        if existing_sha256 != expected_sha256:
            raise ValueError(
                f"Existing baseline clip {clip_index} used a different condition "
                f"asset: {existing_path}"
            )
        condition_hashes[name] = expected_sha256

    video_path = (
        existing_root
        / example
        / args.existing_inference_folder
        / f"video_seed={args.seed}.mp4"
    )
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    video, fps = load_video(video_path)
    if video.shape[0] != args.num_frames:
        raise ValueError(
            f"Existing baseline clip {clip_index} has {video.shape[0]} frames, "
            f"expected {args.num_frames}"
        )
    if tuple(video.shape[1:3]) != (args.height, args.width):
        raise ValueError(
            f"Existing baseline clip {clip_index} has spatial shape "
            f"{video.shape[2]}x{video.shape[1]}, expected "
            f"{args.width}x{args.height} for resolution={args.resolution}"
        )

    log_path = None
    if args.existing_log_dir:
        log_dir = resolve_path(args.existing_log_dir, repo_root())
        matches = sorted(log_dir.glob(f"{clip_index}_{example}_inference.log"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one inference log for clip {clip_index} under {log_dir}, "
                f"found {len(matches)}"
            )
        log_path = matches[0]
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        required_log_fragments = (
            f"EXAMPLE={example}",
            f"SEEDS={args.seed}",
            "50/50",
        )
        missing = [value for value in required_log_fragments if value not in log_text]
        if missing:
            raise ValueError(
                f"Existing baseline log {log_path} is incomplete; missing {missing}"
            )

    record = {
        "schema_version": 1,
        "clip_index": clip_index,
        "example": example,
        "condition_folder": str(expected_condition.resolve()),
        "generation_mode": "trusted_existing_50_step_output",
        "condition_equivalence_sha256": condition_hashes,
        "evidence_log": None if log_path is None else str(log_path.resolve()),
        "contract_sha256": contract_sha256,
        "output_video": str(video_path.resolve()),
        "output_video_sha256": sha256_file(video_path),
        "frames": int(video.shape[0]),
        "fps": float(fps),
        "wall_seconds": None,
        "cuda_peak_allocated_gib": None,
        "cuda_peak_reserved_gib": None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Imported verified baseline clip {clip_index}: {video_path}")
    return record


@torch.no_grad()
def main(args) -> None:
    root = repo_root()
    validate_resolution_dimensions(
        args.resolution,
        height=args.height,
        width=args.width,
    )
    manifest_path = resolve_path(args.manifest, root)
    condition_root = resolve_path(args.condition_root, root)
    output_folder = resolve_path(args.output_folder, root)
    checkpoint_path = resolve_checkpoint(resolve_path(args.vista4d_checkpoint, root)).path
    config_path = resolve_path(args.vista4d_config_path, root)
    if manifest_path.suffix.lower() != ".json":
        raise ValueError("The frozen baseline requires a JSON manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_resolution(
        manifest,
        resolution=args.resolution,
        height=args.height,
        width=args.width,
    )
    clips = validate_flowlong_window_manifest(
        manifest,
        clip_frames=args.num_frames,
        overlap=5,
        temporal_alignment=4,
    )
    if int(manifest["stride"]) != 44:
        raise ValueError(f"Baseline requires stride=44, got {manifest['stride']}")
    if args.num_inference_steps <= 0 or args.cfg_scale < 0 or args.sigma_shift <= 0:
        raise ValueError('Invalid inference steps/CFG/sigma shift')
    if not getattr(args, 'allow_custom_inference', False) and args.num_inference_steps != 50:
        raise ValueError("The formal stage-4 baseline requires 50 inference steps")
    if not getattr(args, 'allow_custom_inference', False) and (args.cfg_scale != 5.0 or args.sigma_shift != 5.0):
        raise ValueError("The formal stage-4 baseline requires CFG=5 and sigma_shift=5")

    checkpoint_sha256 = args.vista4d_checkpoint_sha256
    if checkpoint_sha256 is None:
        checkpoint_sha256 = hash_checkpoint(checkpoint_path)
    checkpoint_sha256 = validate_digest(
        checkpoint_sha256, "--vista4d_checkpoint_sha256"
    )
    contract, condition_entries, shared_render_folder = load_baseline_contract(
        args,
        manifest_path=manifest_path,
        manifest=manifest,
        clips=clips,
        condition_root=condition_root,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    contract_sha256 = canonical_sha256(contract)

    output_folder.mkdir(parents=True, exist_ok=True)
    final_video_path = output_folder / f"video_seed={args.seed}_center_cut.mp4"
    final_report_path = output_folder / f"baseline_report_seed={args.seed}.json"
    if not args.overwrite and final_video_path.is_file() and final_report_path.is_file():
        previous = json.loads(final_report_path.read_text(encoding="utf-8"))
        if (
            previous.get("contract_sha256") == contract_sha256
            and previous.get("output_video_sha256") == sha256_file(final_video_path)
        ):
            print(f"Skip completed frozen baseline: {final_video_path}")
            return
        raise FileExistsError(
            f"Baseline output exists with a different or incomplete contract: "
            f"{final_video_path}. Pass --overwrite to replace it."
        )

    existing_root = (
        resolve_path(args.existing_clip_root, root)
        if args.existing_clip_root
        else None
    )
    clip_records = []
    pending = []
    for clip, condition in zip(clips, condition_entries):
        example = condition["example"]
        clip_folder = output_folder / "clips" / example
        video_path = (
            existing_root
            / example
            / args.existing_inference_folder
            / f"video_seed={args.seed}.mp4"
            if existing_root is not None
            else clip_folder / f"video_seed={args.seed}.mp4"
        )
        report_path = clip_folder / f"clip_report_seed={args.seed}.json"
        existing = None if args.overwrite else completed_clip(
            video_path, report_path, contract_sha256
        )
        if existing is None and existing_root is not None:
            clip_records.append(
                import_existing_clip(
                    args=args,
                    clip=clip,
                    condition=condition,
                    existing_root=existing_root,
                    report_path=report_path,
                    contract_sha256=contract_sha256,
                )
            )
        elif existing is None:
            if not args.overwrite and (video_path.exists() or report_path.exists()):
                raise FileExistsError(
                    f"Partial or mismatched clip output: {clip_folder}. "
                    "Pass --overwrite to replace it."
                )
            pending.append((clip, condition, video_path, report_path))
        else:
            print(f"Skip completed baseline clip {clip['clip_index']}: {video_path}")
            clip_records.append(existing)

    pipe = None
    if pending:
        with config_path.open("r", encoding="utf-8") as file:
            vista4d_config = yaml.safe_load(file)
        args.vista4d_checkpoint = str(checkpoint_path)
        args.vista4d_config_path = str(config_path)
        args.use_usp = False
        args.cfg_merge = False
        args.seed = [int(args.seed)]
        pipe = get_pipeline(args, vista4d_config)

        for clip, condition, video_path, report_path in pending:
            started = perf_counter()
            args.input_folder = condition["folder"]
            inputs, fps = get_inputs(args, vista4d_config)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats("cuda")
            videos = pipe(
                **inputs,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                cfg_merge=False,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                tiled=args.tile_vae,
            )
            frames = np.stack([np.asarray(frame) for frame in videos[0]], axis=0)
            if frames.shape[0] != args.num_frames:
                raise ValueError(
                    f"Baseline clip {clip['clip_index']} produced {frames.shape[0]} "
                    f"frames, expected {args.num_frames}"
                )
            if tuple(frames.shape[1:3]) != (args.height, args.width):
                raise ValueError(
                    f"Baseline clip {clip['clip_index']} produced spatial shape "
                    f"{frames.shape[2]}x{frames.shape[1]}, expected "
                    f"{args.width}x{args.height}"
                )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            save_video(video_path, frames, fps=float(fps), quality=args.quality)
            record = {
                "schema_version": 1,
                "clip_index": int(clip["clip_index"]),
                "example": condition["example"],
                "condition_folder": condition["folder"],
                "contract_sha256": contract_sha256,
                "output_video": str(video_path.resolve()),
                "output_video_sha256": sha256_file(video_path),
                "frames": int(frames.shape[0]),
                "fps": float(fps),
                "wall_seconds": perf_counter() - started,
                "cuda_peak_allocated_gib": (
                    float(torch.cuda.max_memory_allocated("cuda") / (1024**3))
                    if torch.cuda.is_available()
                    else None
                ),
                "cuda_peak_reserved_gib": (
                    float(torch.cuda.max_memory_reserved("cuda") / (1024**3))
                    if torch.cuda.is_available()
                    else None
                ),
            }
            report_path.write_text(
                json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
            )
            clip_records.append(record)
            print(f"Baseline clip {clip['clip_index']}: {video_path}")

    record_by_index = {int(item["clip_index"]): item for item in clip_records}
    ordered_records = [record_by_index[int(clip["clip_index"])] for clip in clips]
    clip_videos = []
    fps_values = []
    for clip, record in zip(clips, ordered_records):
        path = Path(record["output_video"])
        video, fps = load_video(path)
        video = prepare_clip_video(
            video,
            clip,
            strict_num_frames=True,
            path=path,
        )
        clip_videos.append(video)
        fps_values.append(float(fps))
    if max(fps_values) - min(fps_values) > 1e-3:
        raise ValueError(f"Baseline clip FPS values do not match: {fps_values}")

    merged, merge_log = merge_videos(
        clips,
        clip_videos,
        mode="center_cut",
        flow_max_pixels=32.0,
        flow_consistency_sigma=2.5,
    )
    expected_frames = int(manifest["end_exclusive"]) - int(manifest["start_frame"])
    if merged.shape[0] != expected_frames:
        raise ValueError(
            f"Merged baseline has {merged.shape[0]} frames, expected {expected_frames}"
        )
    fps = float(manifest.get("fps", fps_values[0]))
    save_video(final_video_path, merged, fps=fps, quality=args.quality)
    final_report = {
        "schema_version": 1,
        "implementation": "vista4d-independent-center-cut-baseline-v1",
        "contract": contract,
        "contract_sha256": contract_sha256,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": contract["manifest_sha256"],
        },
        "conditions": {
            "root": str(condition_root.resolve()),
            "render_folder": args.render_folder,
            "full_render_folder": shared_render_folder,
            "windows": condition_entries,
        },
        "baseline_source": {
            "mode": (
                "trusted_existing_50_step_output"
                if existing_root is not None
                else "fresh_isolated_generation"
            ),
            "existing_clip_root": (
                None if existing_root is None else str(existing_root.resolve())
            ),
            "existing_log_dir": args.existing_log_dir,
        },
        "model": {
            "model_id_with_origin_paths": args.model_id_with_origin_paths,
            "local_model_folder": args.local_model_folder,
            "vista4d_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "size_bytes": checkpoint_size(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            "vista4d_config_path": str(config_path.resolve()),
        },
        "experiment": {
            "seed": int(args.seed[0]) if isinstance(args.seed, list) else int(args.seed),
            "num_inference_steps": int(args.num_inference_steps),
            "sigma_shift": float(args.sigma_shift),
            "cfg_scale": float(args.cfg_scale),
            "cfg_merge": False,
            "use_usp": False,
            "prompt_sha256": sha256_text(args.prompt),
            "negative_prompt_sha256": sha256_text(args.negative_prompt),
            "window_seed_policy": "same_seed_for_each_independent_window",
        },
        "repository": git_metadata(root),
        "resolution": args.resolution,
        "clips": ordered_records,
        "merge": {
            "mode": "center_cut",
            "log": merge_log,
            "handoff_frames": handoff_frames(merge_log),
        },
        "fps": fps,
        "output_video": str(final_video_path.resolve()),
        "output_video_sha256": sha256_file(final_video_path),
        "output_frames": int(merged.shape[0]),
        "height": int(merged.shape[1]),
        "width": int(merged.shape[2]),
    }
    final_report_path.write_text(
        json.dumps(final_report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Frozen baseline video: {final_video_path}")
    print(f"Frozen baseline report: {final_report_path}")


def default_vram_limit():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 2


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate the isolated 50-step independent center-cut baseline."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--condition_root", required=True)
    parser.add_argument("--render_folder", default="render_384p_smooth")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--existing_clip_root", default=None)
    parser.add_argument(
        "--existing_inference_folder", default="vista4d_384p_smooth"
    )
    parser.add_argument("--existing_log_dir", default=None)
    parser.add_argument("--resolution", default="384p")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)

    parser.add_argument("--model_id_with_origin_paths", required=True)
    parser.add_argument("--tokenizer_id_with_origin_path", required=True)
    parser.add_argument("--local_model_folder", required=True)
    parser.add_argument("--vista4d_checkpoint", required=True)
    parser.add_argument("--vista4d_checkpoint_sha256", default=None)
    parser.add_argument("--vista4d_config_path", required=True)
    # get_pipeline() resolves the VRAM preset, so this entry point needs the same flags as the
    # other inference scripts. Keep the baseline's own --vram_limit default (whole-GPU budget)
    # rather than the preset's None, so 'full' behaves exactly as it did before presets existed.
    add_vram_args(parser)
    parser.set_defaults(vram_limit=default_vram_limit())

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=10027)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument('--allow_custom_inference', action='store_true',
                        help='Allow configured non-baseline steps/CFG/sigma; use isolated outputs')
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--tile_vae", action="store_true")
    parser.add_argument("--quality", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(use_usp=False, cfg_merge=False)
    main(parser.parse_args())
