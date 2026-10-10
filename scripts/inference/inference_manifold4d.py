"""Run Manifold4D on a folder produced by scripts/preprocess/render_single.py."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diffsynth.pipelines.wan_video_manifold4d import Manifold4DPipeline, resize_camera_intrinsics
from diffsynth.utils.vista4d.media import apply_num_frames
from utils.media import load_cameras, load_masks, load_video, np_to_pil, pil_to_np, save_cameras, save_video


NEGATIVE_PROMPT = (
    "gaudy, overexposed, static, blurred details, subtitles, style, artwork, painting, still, "
    "overall gray, worst quality, low quality, JPEG compression, ugly, mutilated, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, "
    "still image, cluttered background, three legs, many people in background, walking backwards"
)


def load_render_inputs(folder):
    folder = Path(folder)
    required = ("video_src.mp4", "video_pc.mp4", "alpha_mask_pc", "cameras_src.npz", "cameras_tgt.npz")
    for name in required:
        if not (folder / name).exists():
            raise FileNotFoundError(f"Manifold4D input requires {folder / name}")
    source, fps = load_video(str(folder / "video_src.mp4"))
    render, render_fps = load_video(str(folder / "video_pc.mp4"))
    if len(source) == 0 or len(source) != len(render) or abs(fps - render_fps) > 1e-3:
        raise ValueError("Source and render must have matching nonempty timelines and FPS")
    src_cameras, src_intrinsics = load_cameras(str(folder / "cameras_src.npz"))
    tgt_cameras, tgt_intrinsics = load_cameras(str(folder / "cameras_tgt.npz"))

    def optional_mask(name):
        return load_masks(str(folder / name)) if (folder / name).is_dir() else None

    inputs = dict(
        source_video=np_to_pil(source), point_cloud_video=np_to_pil(render),
        point_cloud_alpha_mask=load_masks(str(folder / "alpha_mask_pc")),
        source_alpha_mask=optional_mask("alpha_mask_src"),
        source_motion_mask=optional_mask("dynamic_mask_src"),
        point_cloud_motion_mask=optional_mask("dynamic_mask_pc"),
        source_cam_c2w=src_cameras, source_intrinsics=src_intrinsics,
        target_cam_c2w=tgt_cameras, target_intrinsics=tgt_intrinsics,
    )
    count = len(source)
    for name, value in inputs.items():
        if value is not None and len(value) != count:
            raise ValueError(f"{name} has {len(value)} frames; expected {count}")
    for prefix, video in (("source", source), ("point_cloud", render)):
        for suffix in ("alpha_mask", "motion_mask"):
            value = inputs[f"{prefix}_{suffix}"]
            if value is not None and value.shape != video.shape[:3]:
                raise ValueError(f"{prefix}_{suffix} must match its video's [T, H, W] shape")
    return inputs, fps


def add_pipeline_args(parser, checkpoint_required=False):
    parser.add_argument("--wan_checkpoint", type=Path, help="Local Wan-AI/Wan2.1-T2V-14B directory")
    parser.add_argument("--manifold4d_checkpoint", type=Path, required=checkpoint_required,
                        help="Directory containing the three official .pt files")
    parser.add_argument("--config", type=Path, default=Path("configs/manifold4d.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram_preset", choices=("full", "balanced"), default="balanced")
    parser.add_argument("--vram_limit", type=float, default=10.0, help="Offload budget in GiB; balanced mode only")


def load_config(args):
    config = yaml.safe_load(args.config.read_text())
    model = config["model"]
    allowed = {"positional_embedding_offset", "cond_stream_t", "prior_sigma"}
    if set(model) - allowed:
        raise ValueError(f"Unknown Manifold4D model config keys: {sorted(set(model) - allowed)}")
    return config


def get_pipeline(args, *, encode_only=False):
    if args.wan_checkpoint is None:
        raise ValueError("--wan_checkpoint is required to load the local base model")
    config = load_config(args)
    return Manifold4DPipeline.from_pretrained(
        args.wan_checkpoint, args.manifold4d_checkpoint,
        device=args.device, torch_dtype=torch.bfloat16,
        cpu_offload=args.vram_preset == "balanced", vram_limit=args.vram_limit,
        encode_only=encode_only, **config["model"],
    )


def main():
    parser = ArgumentParser(description=__doc__)
    add_pipeline_args(parser)
    parser.add_argument("--input_folder", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    for name in ("height", "width", "num_frames", "num_inference_steps"):
        parser.add_argument("--" + name, type=int)
    for name in ("cfg_scale", "sigma_shift"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--prior_sigma", type=float, help="Explicit prior-strength ablation override")
    parser.add_argument("--solver", choices=("unipc", "euler"))
    parser.add_argument("--cfg_merge", action="store_true")
    parser.add_argument("--tile_vae", action="store_true")
    parser.add_argument("--rand_device", help="Default: --device. cuda matches the official generator; cpu enables portable seeds")
    parser.add_argument("--validate_only", action="store_true", help="Check input timelines/masks/cameras without loading weights")
    args = parser.parse_args()
    config = load_config(args)
    options = {k: getattr(args, k) if getattr(args, k) is not None else v for k, v in config["inference"].items()}
    height, width, frames = (options[k] for k in ("height", "width", "num_frames"))
    if min(height, width) < 16 or height % 16 or width % 16 or frames < 1 or (frames - 1) % 4:
        parser.error("Require positive 16-aligned dimensions and 4n+1 frames")
    inputs, fps = load_render_inputs(args.input_folder)
    if len(inputs["source_video"]) < frames:
        parser.error("Input video is shorter than num_frames")
    # Validate and exercise the same ray construction used by inference.
    from diffsynth.models.wan_video_manifold4d import camera_tokens
    for prefix, video_name in (("source", "source_video"), ("target", "point_cloud_video")):
        video = inputs[video_name]
        size = (video[0].height, video[0].width)
        intrinsics = resize_camera_intrinsics(apply_num_frames(inputs[prefix + "_intrinsics"], frames), size, (height, width))
        cameras = apply_num_frames(inputs[prefix + "_cam_c2w"], frames)
        camera_tokens(torch.tensor(cameras)[None], torch.tensor(intrinsics)[None],
                      ((frames - 1) // 4 + 1, height // 16, width // 16), (height, width))
    if args.validate_only:
        print(json.dumps({"valid": True, "input_frames": len(inputs["source_video"]), "fps": fps,
                          "output_shape": [frames, height, width], "model_loaded": False}, indent=2))
        return
    if args.manifold4d_checkpoint is None or args.output_folder is None:
        parser.error("Inference requires --manifold4d_checkpoint and --output_folder")
    pipe = get_pipeline(args)
    if args.prior_sigma is not None:
        pipe.prior_sigma = args.prior_sigma
    videos = pipe(**inputs, **options, prompt=args.prompt, negative_prompt=args.negative_prompt,
                  seed=args.seed, cfg_merge=args.cfg_merge, tiled=args.tile_vae, rand_device=args.rand_device)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    for video, seed in zip(videos, args.seed):
        save_video(str(args.output_folder / f"video_seed={seed}.mp4"), pil_to_np(video), fps=fps, quality=9)
    for prefix, video_name in (("source", "source_video"), ("target", "point_cloud_video")):
        video = inputs[video_name]
        intrinsics = resize_camera_intrinsics(apply_num_frames(inputs[prefix + "_intrinsics"], frames),
                                              (video[0].height, video[0].width), (height, width))
        save_cameras(str(args.output_folder / f"cameras_{prefix}.npz"),
                     apply_num_frames(inputs[prefix + "_cam_c2w"], frames), intrinsics)
    metadata = {"method": "Manifold4D", "geometry_frontend": "existing repository render",
                "upstream_commit": "14e1989e0e854a1c05eae0cf09533a7558adf20b",
                "wan_checkpoint": str(args.wan_checkpoint.resolve()),
                "manifold4d_checkpoint": str(args.manifold4d_checkpoint.resolve()),
                "input_folder": str(args.input_folder.resolve()), "prompt": args.prompt,
                "negative_prompt": args.negative_prompt, "seed": args.seed, "fps": fps,
                "prior_sigma": pipe.prior_sigma, "model": {**config["model"], "prior_sigma": pipe.prior_sigma}, "inference": options,
                "rand_device": args.rand_device or args.device, "cfg_merge": args.cfg_merge, "tile_vae": args.tile_vae,
                "vram_preset": args.vram_preset, "vram_limit": args.vram_limit}
    (args.output_folder / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
