from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from os import path
from pathlib import Path
import time

import numpy as np
import torch

from utils.media import (
    concat_2d, crop_and_resize_video, depths_to_disparity_video, load_video, mask_to_video,
    save_cameras, save_clips, save_depths, save_masks, save_mp4_with_gif, save_video, slice_center_frames,
)
from utils.misc import cleanup


_GIB = 1024 ** 3


def _cuda_snapshot():
    """Synchronize CUDA and capture allocator plus device-level memory state."""
    if not torch.cuda.is_available():
        return None

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / _GIB,
        "reserved_gib": torch.cuda.memory_reserved() / _GIB,
        "device_free_gib": free_bytes / _GIB,
        "device_total_gib": total_bytes / _GIB,
    }


def _start_cuda_phase(enabled):
    if not enabled:
        return None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return {
        "started_at": time.perf_counter(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_memory": _cuda_snapshot(),
    }


def _finish_cuda_phase(state):
    if state is None:
        return None

    torch.cuda.synchronize()
    result = {
        "wall_seconds": time.perf_counter() - state["started_at"],
        "started_at_utc": state["started_at_utc"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_memory": state["start_memory"],
        "end_memory": _cuda_snapshot(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / _GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / _GIB,
    }
    return result


@torch.no_grad()
def main(args):
    profiling_enabled = args.profile_json is not None
    total_started_at = time.perf_counter()
    profile = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_args": vars(args).copy(),
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
        },
        "phases": {},
    }

    # Load source video (center-sliced to args.num_frames, cropped/resized to target resolution)
    input_started_at = time.perf_counter()
    video_src, fps = load_video(args.video_path)
    num_src_input_frames, height_src_input, width_src_input, _ = video_src.shape
    assert num_src_input_frames >= args.num_frames,\
        f"Source video must have at least args.num_frames={args.num_frames} frames, got {num_src_input_frames} frames."

    video_src = slice_center_frames(video_src, args.num_frames)
    video_src = crop_and_resize_video(video_src, args.height, args.width, resample="lanczos")
    print(
        f"Sliced and resized source video from {num_src_input_frames}x{width_src_input}x{height_src_input} "
        f"to {args.num_frames}x{args.width}x{args.height}",
    )

    # Optionally load DSE (dynamic scene expansion) casual scene capture, subsample evenly, and concatenate
    clips = None
    if args.dse_video_path is not None:
        video_dse, _ = load_video(args.dse_video_path)  # DSE fps ignored, combined video uses source fps
        num_dse_input_frames, height_dse_input, width_dse_input, _ = video_dse.shape
        num_dse_frames = args.num_dse_frames if args.num_dse_frames is not None else num_dse_input_frames
        assert num_dse_input_frames >= num_dse_frames,\
            f"DSE video must have at least num_dse_frames={num_dse_frames} frames, got {num_dse_input_frames} frames."

        dse_indices = np.linspace(0, num_dse_input_frames - 1, num_dse_frames).round().astype(int)
        video_dse = video_dse[dse_indices]
        video_dse = crop_and_resize_video(video_dse, args.height, args.width, resample="lanczos")
        print(
            f"Subsampled and resized DSE video from {num_dse_input_frames}x{width_dse_input}x{height_dse_input} "
            f"to {num_dse_frames}x{args.width}x{args.height}",
        )
        video = np.concatenate([video_src, video_dse], axis=0)
        clips = {"src": (0, args.num_frames), "dse": (args.num_frames, args.num_frames + num_dse_frames)}
    else:
        video = video_src
    profile["phases"]["input_decode_and_resize"] = {
        "wall_seconds": time.perf_counter() - input_started_at,
    }

    if args.recon_method == "da3":  # 4D reconstruction (and sky mask segmentation) with Depth Anything 3
        from utils.recon_and_seg.recon_da3 import init_da3, run_da3

        recon_phase = _start_cuda_phase(profiling_enabled)
        model_load_started_at = time.perf_counter()
        da3_model = init_da3(model_id=args.da3_model_id, device="cuda")
        if profiling_enabled:
            torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_load_started_at

        inference_started_at = time.perf_counter()
        depths, sky_mask, cam_c2w, intrinsics =\
            run_da3(video, model=da3_model, process_res=args.da3_process_res, resize_output=True)
        if profiling_enabled:
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started_at
        recon_profile = _finish_cuda_phase(recon_phase)
        if recon_profile is not None:
            recon_profile.update({
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
            })
            profile["phases"]["depth_anything_3"] = recon_profile
        del da3_model

    elif args.recon_method == "pi3":
        from utils.recon_and_seg.recon_pi3 import init_pi3, run_pi3

        recon_phase = _start_cuda_phase(profiling_enabled)
        model_load_started_at = time.perf_counter()
        pi3_model = init_pi3(model_id=args.pi3_model_id, device="cuda")
        if profiling_enabled:
            torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_load_started_at

        inference_started_at = time.perf_counter()
        depths, cam_c2w, intrinsics = run_pi3(
            video, model=pi3_model, pixel_limit=args.pi3_pixel_limit, resize_output=True,
            dtype=torch.bfloat16, device="cuda", head_chunk_size=args.pi3_head_chunk_size,
        )
        if profiling_enabled:
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started_at
        recon_profile = _finish_cuda_phase(recon_phase)
        if recon_profile is not None:
            recon_profile.update({
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
            })
            profile["phases"]["pi3"] = recon_profile
        del pi3_model

    if args.scene_scale != 1.0:
        depths *= args.scene_scale
        cam_c2w[:, :3, 3] *= args.scene_scale

    cleanup()
    if profiling_enabled:
        recon_key = "depth_anything_3" if args.recon_method == "da3" else "pi3"
        profile["phases"][recon_key]["after_cleanup_memory"] = _cuda_snapshot()

    # Dynamic mask segmentation with Segment Anything 3
    num_frames, height, width, _ = video.shape
    if len(args.seg_keywords) == 0:
        dynamic_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
        sky_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    elif "_all_" in args.seg_keywords:
        dynamic_mask = np.ones((num_frames, height, width), dtype=np.bool_)
        sky_mask = np.zeros((num_frames, height, width), dtype=np.bool_)

    elif args.sam3_implementation == "official":
        from utils.recon_and_seg.seg_sam3_official import init_sam3_video, run_sam3_video

        sam3_phase = _start_cuda_phase(profiling_enabled)
        model_load_started_at = time.perf_counter()
        sam3_video_predictor = init_sam3_video()
        if profiling_enabled:
            torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_load_started_at

        inference_started_at = time.perf_counter()
        dynamic_mask, seg_frames = run_sam3_video(video, sam3_video_predictor, args.seg_keywords)
        if args.recon_method == "pi3":  # Unlike DA3, Pi3 doesn't segment the sky, so we have to do it with SAM3
            sky_mask, _ = run_sam3_video(video, sam3_video_predictor, ["sky"])
        if profiling_enabled:
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started_at
        sam3_profile = _finish_cuda_phase(sam3_phase)
        if sam3_profile is not None:
            sam3_profile.update({
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
            })
            profile["phases"]["sam3"] = sam3_profile
        del sam3_video_predictor

    elif args.sam3_implementation == "transformers":
        # Requires transformers >= 5.0.0, which xfuser==0.45.0 doesn't support
        from utils.recon_and_seg.seg_sam3_transformers import init_sam3_video, run_sam3_video
        dtype = torch.bfloat16
        device = "cuda"
        sam3_phase = _start_cuda_phase(profiling_enabled)
        model_load_started_at = time.perf_counter()
        sam3_model, sam3_processor = init_sam3_video(args.sam3_model_id, dtype=dtype, device=device)
        if profiling_enabled:
            torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_load_started_at

        inference_started_at = time.perf_counter()
        dynamic_mask, seg_frames =\
            run_sam3_video(video, sam3_model, sam3_processor, args.seg_keywords, dtype=dtype, device=device)
        if args.recon_method == "pi3":  # Unlike DA3, Pi3 doesn't segment the sky, so we have to do it with SAM3
            sky_mask, _ = run_sam3_video(video, sam3_model, sam3_processor, ["sky"], dtype=dtype, device=device)
        if profiling_enabled:
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started_at
        sam3_profile = _finish_cuda_phase(sam3_phase)
        if sam3_profile is not None:
            sam3_profile.update({
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
            })
            profile["phases"]["sam3"] = sam3_profile
        del sam3_model, sam3_processor

    cleanup()
    if profiling_enabled and "sam3" in profile["phases"]:
        profile["phases"]["sam3"]["after_cleanup_memory"] = _cuda_snapshot()

    # Save results
    save_started_at = time.perf_counter()
    save_video(path.join(args.output_folder, "video.mp4"), video, fps=fps, quality=9)
    save_depths(path.join(args.output_folder, "depths"), depths, dtype=np.float16)
    save_cameras(path.join(args.output_folder, "cameras.npz"), cam_c2w, intrinsics)
    save_masks(path.join(args.output_folder, "dynamic_mask"), dynamic_mask)
    save_masks(path.join(args.output_folder, "sky_mask"), sky_mask)
    if clips is not None:
        save_clips(path.join(args.output_folder, "clips.json"), clips)

    if args.save_vis:  # Save visualization as video
        depths_video = depths_to_disparity_video(depths, ~sky_mask, use_colormap=True)
        sky_mask_video = mask_to_video(sky_mask)

        if len(args.seg_keywords) == 0 or "_all_" in args.seg_keywords:
            seg_overlay_video = video
        else:
            from utils.recon_and_seg.seg_sam3_utils import overlay_instances_on_video
            seg_overlay_video = overlay_instances_on_video(video, seg_frames, alpha=0.5)
        dynamic_mask_video = mask_to_video(dynamic_mask)

        vis_video = concat_2d([[depths_video, sky_mask_video], [seg_overlay_video, dynamic_mask_video]])
        save_mp4_with_gif(
            path.join(args.output_folder, "vis"), vis_video, fps=fps, quality=5,
            gif_folder="" if args.save_vis_gif else None,
        )

    profile["phases"]["save_outputs"] = {
        "wall_seconds": time.perf_counter() - save_started_at,
    }
    profile["total_wall_seconds"] = time.perf_counter() - total_started_at
    if profiling_enabled:
        profile_path = Path(args.profile_json)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w", encoding="utf-8") as profile_file:
            json.dump(profile, profile_file, indent=2)
            profile_file.write("\n")
        print(f"Profiling results saved to: {profile_path}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--dse_video_path", default=None, type=str)  # Optional casual scene capture for DSE
    parser.add_argument("--output_folder", required=True, type=str)
    parser.add_argument("--seg_keywords", type=str, nargs="*", default=["_all_"])

    parser.add_argument("--save_vis", action="store_true", default=False)
    parser.add_argument("--save_vis_gif", action="store_true", default=False)
    parser.add_argument(
        "--profile_json",
        default=None,
        type=str,
        help=(
            "Optional JSON path for per-phase wall-clock and CUDA memory profiling. "
            "CUDA synchronization is enabled only when this is set."
        ),
    )

    parser.add_argument("--recon_method", default="da3", type=str, choices=("pi3", "da3"))
    parser.add_argument("--pi3_model_id", default="yyfz233/Pi3X", type=str)
    parser.add_argument("--da3_model_id", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1", type=str)
    parser.add_argument("--sam3_implementation", default="official", choices=("official", "transformers"))
    parser.add_argument("--sam3_model_id", default="facebook/sam3", type=str)

    parser.add_argument("--height", default=720, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--num_frames", default=121, type=int)
    parser.add_argument("--num_dse_frames", default=None, type=int)  # None = use all DSE input frames
    parser.add_argument("--pi3_pixel_limit", default=255000, type=int)
    parser.add_argument("--pi3_head_chunk_size", default=16, type=int)
    parser.add_argument("--da3_process_res", default=-1, type=int)
    parser.add_argument("--scene_scale", default=1.0, type=float)  # Scale the depths and cam_c2w of 4D reconstruction

    args = parser.parse_args()

    if args.pi3_pixel_limit <= 0:  # Process Pi3(X) at maximum (video) resolution
        args.pi3_pixel_limit = args.height * args.width
    if args.da3_process_res <= 0:  # Process DA3 at maximum (video) resolution
        args.da3_process_res = args.width

    main(args)
