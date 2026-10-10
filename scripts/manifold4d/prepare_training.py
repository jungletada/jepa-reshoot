"""Cache supervised aligned pairs; target video is used only as a VAE teacher."""
from argparse import ArgumentParser
import json
from pathlib import Path

import torch

from diffsynth.pipelines.manifold4d import pool_video_mask
from diffsynth.utils.vista4d.media import apply_num_frames, crop_and_resize_tensor
from scripts.inference.inference_manifold4d import add_pipeline_args, get_pipeline, load_render_inputs
from utils.media import load_masks, load_video, np_to_pil


def main():
    parser = ArgumentParser(description=__doc__)
    add_pipeline_args(parser)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSONL: input_folder, target_video, prompt; optional target_motion_mask directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--tile_vae", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or any(not all(k in row for k in ("input_folder", "target_video", "prompt")) for row in rows):
        parser.error("Each manifest row needs input_folder, target_video and prompt")
    if args.output.exists():
        raise FileExistsError(f"Use a new cache directory: {args.output}")
    pipe = get_pipeline(args, encode_only=True)
    args.output.mkdir(parents=True)
    index = []
    for i, row in enumerate(rows):
        folder = args.manifest.parent / row["input_folder"]
        target_path = args.manifest.parent / row["target_video"]
        inputs, fps = load_render_inputs(folder)
        target, target_fps = load_video(str(target_path))
        if len(target) != len(inputs["source_video"]) or abs(fps - target_fps) > 1e-3:
            raise ValueError("Target teacher and source must share the timeline and FPS")
        example = pipe.prepare_inputs(**inputs, height=args.height, width=args.width,
                                      num_frames=args.num_frames, tiled=args.tile_vae)
        example["target_latents"] = pipe.encode_video(np_to_pil(target), args.height, args.width,
                                                      args.num_frames, args.tile_vae)
        example["context"] = pipe.encode_prompt([row["prompt"]])
        example["empty_context"] = pipe.encode_prompt([""])
        if row.get("target_motion_mask"):
            motion = load_masks(str(args.manifest.parent / row["target_motion_mask"]))
            if motion.shape != target.shape[:3]:
                raise ValueError("Target motion mask must match the teacher video's [T,H,W] grid")
            motion = torch.tensor(apply_num_frames(motion, args.num_frames), device=pipe.device, dtype=torch.float32)[None, None]
            motion = crop_and_resize_tensor(motion, args.height, args.width, mode="bilinear")
            example["target_motion_mask"] = pool_video_mask(motion)
        example["metadata"] = {"method": "Manifold4D", "wan_checkpoint": str(args.wan_checkpoint.resolve()),
                               "input_folder": str(folder.resolve()), "target_video": str(target_path.resolve()),
                               "prompt": row["prompt"], "fps": fps, "shape": [args.num_frames, args.height, args.width]}
        from diffsynth.pipelines.manifold4d_training import validate_training_example
        validate_training_example(example)
        filename = f"pair_{i:06d}.pt"
        torch.save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in example.items()}, args.output / filename)
        index.append({"cache": filename})
        print(json.dumps({"cached_pair": i + 1, "file": filename}), flush=True)
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
    pipe.load_models_to_device([])


if __name__ == "__main__":
    main()
