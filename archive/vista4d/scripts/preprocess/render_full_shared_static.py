from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from utils.media import (
    intrinsics_to_K,
    load_cameras,
    load_recon_and_seg,
    resize_intrinsics,
    save_cameras,
    save_depths,
    save_masks,
)
from utils.misc import cleanup
from utils.point_cloud.point_cloud import render_frame, unproject
from utils.point_cloud.preprocess import SKY_DEPTH, preprocess_scene


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def prepare_output_folder(folder: Path, overwrite: bool) -> None:
    if folder.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {folder}. Pass --overwrite to replace it."
            )
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def save_lossless_rgb_video(path: Path, video: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264rgb",
        pixelformat="rgb24",
        ffmpeg_params=["-crf", "0", "-preset", "medium"],
        macro_block_size=1,
    )
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()


def static_frame_indices(num_frames: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"static_frame_stride must be positive, got {stride}")
    indices = list(range(0, num_frames, stride))
    if indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    return np.asarray(indices, dtype=np.int64)


def to_gpu_scene(processed: Dict, device: str) -> Dict[str, torch.Tensor]:
    dtype = torch.float32
    flip = torch.diag(
        torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=dtype, device=device)
    )
    cam_c2w = torch.from_numpy(processed["cam_c2w"]).to(dtype=dtype, device=device)
    return {
        "video": torch.from_numpy(processed["video"]).to(dtype=dtype, device=device) / 255.0,
        "depths": torch.from_numpy(processed["depths"]).to(dtype=dtype, device=device),
        "cam_c2w": flip[None] @ cam_c2w,
        "K": torch.from_numpy(intrinsics_to_K(processed["intrinsics"])).to(
            dtype=dtype, device=device
        ),
        "dynamic_mask": torch.from_numpy(processed["dynamic_mask"]).to(
            dtype=torch.bool, device=device
        ),
        "static_mask": torch.from_numpy(processed["static_mask"]).to(
            dtype=torch.bool, device=device
        ),
    }


@torch.no_grad()
def build_static_point_bank(
    scene: Dict,
    indices: np.ndarray,
    height: int,
    width: int,
    depth_outliers: str,
    ignore_sky_mask: bool,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    color_parts = []
    position_parts = []
    counts = []
    for frame_index in tqdm(indices, desc="Building shared static point bank"):
        processed = preprocess_scene(
            scene,
            indices=np.asarray([frame_index]),
            height=height,
            width=width,
            depth_outliers=depth_outliers,
            ignore_sky_mask=ignore_sky_mask,
        )
        gpu = to_gpu_scene(processed, device=device)
        zero_dynamic = torch.zeros_like(gpu["static_mask"])
        colors, positions, _, _ = unproject(
            video=gpu["video"],
            depths=gpu["depths"],
            cam_c2w=gpu["cam_c2w"],
            K=gpu["K"],
            dynamic_mask=zero_dynamic,
            static_mask=gpu["static_mask"],
        )
        color_parts.append(colors)
        position_parts.append(positions)
        counts.append(int(colors.shape[0]))
        del processed, gpu, zero_dynamic

    if not color_parts or sum(counts) == 0:
        raise ValueError("No static points were produced from the selected source frames")
    colors = torch.cat(color_parts, dim=0)
    positions = torch.cat(position_parts, dim=0)
    del color_parts, position_parts
    cleanup()
    return colors, positions, counts


def allocate_outputs(num_frames: int, height: int, width: int) -> Dict[str, np.ndarray]:
    return {
        "video_src": np.empty((num_frames, height, width, 3), dtype=np.uint8),
        "video_pc": np.empty((num_frames, height, width, 3), dtype=np.uint8),
        "depths_src": np.empty((num_frames, height, width), dtype=np.float32),
        "depths_pc": np.empty((num_frames, height, width), dtype=np.float32),
        "alpha_mask_src": np.ones((num_frames, height, width), dtype=np.bool_),
        "alpha_mask_pc": np.empty((num_frames, height, width), dtype=np.bool_),
        "dynamic_mask_src": np.empty((num_frames, height, width), dtype=np.bool_),
        "dynamic_mask_pc": np.empty((num_frames, height, width), dtype=np.bool_),
        "static_mask_src": np.empty((num_frames, height, width), dtype=np.bool_),
        "static_mask_pc": np.empty((num_frames, height, width), dtype=np.bool_),
        "sky_mask_src": np.empty((num_frames, height, width), dtype=np.bool_),
        "cameras_src": np.empty((num_frames, 4, 4), dtype=np.float32),
        "intrinsics_src": np.empty((num_frames, 4), dtype=np.float32),
    }


@torch.no_grad()
def render_target_chunks(
    scene: Dict,
    static_colors: torch.Tensor,
    static_positions: torch.Tensor,
    target_c2w: np.ndarray,
    target_intrinsics: np.ndarray,
    global_indices: np.ndarray,
    height: int,
    width: int,
    chunk_size: int,
    depth_outliers: str,
    ignore_sky_mask: bool,
    device: str,
) -> Dict[str, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError(f"render_chunk_size must be positive, got {chunk_size}")
    outputs = allocate_outputs(len(global_indices), height=height, width=width)
    static_dynamic_labels = torch.zeros(
        static_colors.shape[0], dtype=torch.bool, device=device
    )
    flip = torch.diag(
        torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    )

    for chunk_start in range(0, len(global_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(global_indices))
        source_indices = global_indices[chunk_start:chunk_end]
        processed = preprocess_scene(
            scene,
            indices=source_indices,
            height=height,
            width=width,
            depth_outliers=depth_outliers,
            ignore_sky_mask=ignore_sky_mask,
        )
        chunk_slice = slice(chunk_start, chunk_end)
        outputs["video_src"][chunk_slice] = processed["video"]
        outputs["depths_src"][chunk_slice] = np.minimum(processed["depths"], SKY_DEPTH)
        outputs["dynamic_mask_src"][chunk_slice] = processed["dynamic_mask"]
        outputs["static_mask_src"][chunk_slice] = processed["static_mask"]
        outputs["sky_mask_src"][chunk_slice] = processed["sky_mask"]
        outputs["cameras_src"][chunk_slice] = processed["cam_c2w"]
        outputs["intrinsics_src"][chunk_slice] = processed["intrinsics"]

        gpu = to_gpu_scene(processed, device=device)
        target_c2w_chunk = torch.from_numpy(target_c2w[chunk_slice]).to(
            dtype=torch.float32, device=device
        )
        target_c2w_chunk = flip[None] @ target_c2w_chunk
        target_K_chunk = torch.from_numpy(
            intrinsics_to_K(target_intrinsics[chunk_slice])
        ).to(dtype=torch.float32, device=device)

        progress = tqdm(
            range(chunk_end - chunk_start),
            desc=f"Rendering targets {chunk_start}..{chunk_end - 1}",
            leave=False,
        )
        for local_index in progress:
            dynamic_mask = gpu["dynamic_mask"][local_index : local_index + 1]
            zero_static = torch.zeros_like(dynamic_mask)
            dynamic_colors, dynamic_positions, _, _ = unproject(
                video=gpu["video"][local_index : local_index + 1],
                depths=gpu["depths"][local_index : local_index + 1],
                cam_c2w=gpu["cam_c2w"][local_index : local_index + 1],
                K=gpu["K"][local_index : local_index + 1],
                dynamic_mask=dynamic_mask,
                static_mask=zero_static,
            )
            points_color = torch.cat((static_colors, dynamic_colors), dim=0)
            points_pos = torch.cat((static_positions, dynamic_positions), dim=0)
            dynamic_labels = torch.cat(
                (
                    static_dynamic_labels,
                    torch.ones(
                        dynamic_colors.shape[0], dtype=torch.bool, device=device
                    ),
                ),
                dim=0,
            )
            rgb, depth, alpha, dynamic = render_frame(
                points_color=points_color,
                points_pos=points_pos,
                cam_c2w=target_c2w_chunk[local_index],
                K=target_K_chunk[local_index],
                height=height,
                width=width,
                dynamic_mask=dynamic_labels,
            )
            output_index = chunk_start + local_index
            outputs["video_pc"][output_index] = (
                rgb.cpu().numpy().clip(0.0, 1.0) * 255.0
            ).astype(np.uint8)
            outputs["depths_pc"][output_index] = np.minimum(
                depth.cpu().numpy(), SKY_DEPTH
            )
            outputs["alpha_mask_pc"][output_index] = alpha.cpu().numpy()
            outputs["dynamic_mask_pc"][output_index] = dynamic.cpu().numpy()
            outputs["static_mask_pc"][output_index] = (
                outputs["alpha_mask_pc"][output_index]
                & ~outputs["dynamic_mask_pc"][output_index]
            )
            del (
                dynamic_mask,
                zero_static,
                dynamic_colors,
                dynamic_positions,
                points_color,
                points_pos,
                dynamic_labels,
                rgb,
                depth,
                alpha,
                dynamic,
            )

        del processed, gpu, target_c2w_chunk, target_K_chunk
        cleanup()
    return outputs


def save_outputs(
    output_folder: Path,
    outputs: Dict[str, np.ndarray],
    target_c2w: np.ndarray,
    target_intrinsics: np.ndarray,
    fps: float,
) -> None:
    save_lossless_rgb_video(output_folder / "video_src.mp4", outputs["video_src"], fps)
    save_lossless_rgb_video(output_folder / "video_pc.mp4", outputs["video_pc"], fps)
    save_depths(
        str(output_folder / "depths_src"), outputs["depths_src"], dtype=np.float16
    )
    save_depths(
        str(output_folder / "depths_pc"), outputs["depths_pc"], dtype=np.float16
    )
    for name in (
        "alpha_mask_src",
        "alpha_mask_pc",
        "dynamic_mask_src",
        "dynamic_mask_pc",
        "static_mask_src",
        "static_mask_pc",
        "sky_mask_src",
    ):
        save_masks(str(output_folder / name), outputs[name])
    save_cameras(
        str(output_folder / "cameras_src.npz"),
        outputs["cameras_src"],
        outputs["intrinsics_src"],
    )
    save_cameras(
        str(output_folder / "cameras_tgt.npz"), target_c2w, target_intrinsics
    )


def main(args) -> None:
    root = repo_root()
    recon_folder = resolve_path(args.recon_and_seg_folder, root)
    camera_path = resolve_path(args.cam_path, root)
    output_folder = resolve_path(args.output_folder, root)
    prepare_output_folder(output_folder, overwrite=args.overwrite)

    scene = load_recon_and_seg(str(recon_folder), depths_dtype=np.float16)
    num_source_frames = int(scene["video"].shape[0])
    start = args.start_frame
    end = (
        num_source_frames
        if args.max_frames is None
        else min(num_source_frames, start + args.max_frames)
    )
    if start < 0 or start >= end:
        raise ValueError(f"Invalid target range [{start}, {end})")
    global_indices = np.arange(start, end, dtype=np.int64)

    target_c2w_all, target_intrinsics_all = load_cameras(str(camera_path))
    if len(target_c2w_all) < end:
        raise ValueError(
            f"Target camera has {len(target_c2w_all)} frames, needs at least {end}"
        )
    height_input, width_input = scene["video"].shape[1:3]
    target_c2w = np.asarray(target_c2w_all[start:end], dtype=np.float32)
    target_intrinsics = resize_intrinsics(
        np.asarray(target_intrinsics_all[start:end], dtype=np.float32),
        height=args.height,
        width=args.width,
        height_input=height_input,
        width_input=width_input,
        crop=True,
    )

    torch.cuda.reset_peak_memory_stats(args.device)
    static_indices = static_frame_indices(
        num_source_frames, stride=args.static_frame_stride
    )
    print(
        f"Shared static bank: {len(static_indices)} / {num_source_frames} source frames, "
        f"stride={args.static_frame_stride}"
    )
    static_colors, static_positions, static_counts = build_static_point_bank(
        scene,
        indices=static_indices,
        height=args.height,
        width=args.width,
        depth_outliers=args.depth_outliers,
        ignore_sky_mask=args.ignore_sky_mask,
        device=args.device,
    )
    print(
        f"Shared static points: {len(static_colors):,}; "
        f"GPU allocated={torch.cuda.memory_allocated(args.device) / (1024 ** 3):.2f} GiB"
    )

    outputs = render_target_chunks(
        scene,
        static_colors=static_colors,
        static_positions=static_positions,
        target_c2w=target_c2w,
        target_intrinsics=target_intrinsics,
        global_indices=global_indices,
        height=args.height,
        width=args.width,
        chunk_size=args.render_chunk_size,
        depth_outliers=args.depth_outliers,
        ignore_sky_mask=args.ignore_sky_mask,
        device=args.device,
    )
    save_outputs(
        output_folder,
        outputs=outputs,
        target_c2w=target_c2w,
        target_intrinsics=target_intrinsics,
        fps=float(scene["fps"]),
    )

    report = {
        "source": "full_sequence_shared_static_point_bank",
        "recon_and_seg_folder": str(recon_folder),
        "cam_path": str(camera_path),
        "output_folder": str(output_folder),
        "source_frames": num_source_frames,
        "global_start_frame": start,
        "global_end_exclusive": end,
        "rendered_frames": len(global_indices),
        "height": args.height,
        "width": args.width,
        "static_frame_stride": args.static_frame_stride,
        "static_frame_indices": static_indices.tolist(),
        "static_point_counts": static_counts,
        "static_points_total": int(len(static_colors)),
        "render_chunk_size": args.render_chunk_size,
        "cuda_peak_memory_allocated_gib": float(
            torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
        ),
        "depth_outliers": args.depth_outliers,
        "ignore_sky_mask": args.ignore_sky_mask,
        "video_codec": "lossless libx264rgb",
    }
    with (output_folder / "shared_static_render.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)
    print(f"Full shared-static render saved to: {output_folder}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Render a full target trajectory in chunks using one shared static point "
            "bank and frame-local dynamic points."
        )
    )
    parser.add_argument("--recon_and_seg_folder", required=True)
    parser.add_argument("--cam_path", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--static_frame_stride", type=int, default=4)
    parser.add_argument("--render_chunk_size", type=int, default=4)
    parser.add_argument(
        "--depth_outliers", choices=("gaussian", "pool"), default="gaussian"
    )
    parser.add_argument("--ignore_sky_mask", action="store_true", default=False)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true", default=False)
    main(parser.parse_args())
