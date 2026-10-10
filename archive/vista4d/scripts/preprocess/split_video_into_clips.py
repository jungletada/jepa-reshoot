from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np

from utils.resolution import validate_resolution_dimensions


def resolve_project_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        repo_root / path,
        repo_root.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def center_crop_box(width: int, height: int, target_width: int, target_height: int) -> Tuple[int, int, int, int]:
    src_aspect = width / height
    target_aspect = target_width / target_height

    if src_aspect > target_aspect:
        crop_height = height
        crop_width = round(height * target_aspect)
        x0 = (width - crop_width) // 2
        y0 = 0
    else:
        crop_width = width
        crop_height = round(width / target_aspect)
        x0 = 0
        y0 = (height - crop_height) // 2

    return x0, y0, crop_width, crop_height


def build_clip_starts(
    start_frame: int,
    end_exclusive: int,
    clip_frames: int,
    overlap: int,
    include_tail: bool,
    temporal_alignment: int = 4,
) -> List[int]:
    if clip_frames <= 0:
        raise ValueError(f"--clip_frames must be positive, got {clip_frames}")
    if overlap < 0:
        raise ValueError(f"--overlap must be >= 0, got {overlap}")
    if overlap >= clip_frames:
        raise ValueError(f"--overlap must be smaller than --clip_frames, got {overlap} >= {clip_frames}")
    if end_exclusive <= start_frame:
        raise ValueError(
            f"Need at least one frame from start_frame={start_frame}, "
            f"but end_exclusive={end_exclusive}."
        )
    if temporal_alignment <= 0:
        raise ValueError(
            f"--temporal_alignment must be positive, got {temporal_alignment}"
        )

    stride = clip_frames - overlap
    if start_frame % temporal_alignment != 0:
        raise ValueError(
            f"--start_frame={start_frame} must be divisible by "
            f"--temporal_alignment={temporal_alignment}"
        )
    if stride % temporal_alignment != 0:
        raise ValueError(
            f"stride={stride} (clip_frames={clip_frames} - overlap={overlap}) "
            f"must be divisible by --temporal_alignment={temporal_alignment}"
        )
    if (clip_frames - 1) % temporal_alignment != 0:
        raise ValueError(
            f"clip_frames={clip_frames} must satisfy "
            f"(clip_frames - 1) % temporal_alignment == 0"
        )

    last_start = end_exclusive - clip_frames
    if not include_tail:
        starts = list(range(start_frame, last_start + 1, stride))
        if not starts:
            raise ValueError(
                f"No complete {clip_frames}-frame clips fit in "
                f"[{start_frame}, {end_exclusive}) with tail disabled."
            )
        return starts

    # Never shift the final window away from the regular stride grid. If the
    # last window extends beyond the source range, it is padded on the right.
    starts = [start_frame]
    while starts[-1] + clip_frames < end_exclusive:
        starts.append(starts[-1] + stride)
    return starts


def save_video(output_path: Path, frames: List[np.ndarray], fps: float, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        quality=quality,
        input_params=["-r", format(float(fps), ".12g")],
        macro_block_size=1,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def read_and_write_clips(
    input_path: Path,
    output_dir: Path,
    clip_starts: List[int],
    clip_frames: int,
    source_end_exclusive: int,
    target_width: int,
    target_height: int,
    quality: int,
    source_stem: Optional[str] = None,
) -> Dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, crop_width, crop_height = center_crop_box(src_width, src_height, target_width, target_height)

    source_stem = source_stem or input_path.stem
    clip_rows = []
    for clip_idx, start in enumerate(clip_starts):
        valid_end_exclusive = min(
            start + clip_frames,
            source_end_exclusive,
            total_frames,
        )
        valid_num_frames = valid_end_exclusive - start
        if valid_num_frames <= 0:
            raise ValueError(
                f"Clip {clip_idx} starts outside the source video: "
                f"start={start}, total_frames={total_frames}"
            )
        end = valid_end_exclusive - 1
        pad_right = clip_frames - valid_num_frames
        padded_end = start + clip_frames - 1
        output_name = (
            f"{source_stem}_split{clip_idx:03d}_"
            f"frames{start:06d}_{end:06d}_{target_height}p{clip_frames}.mp4"
        )
        output_path = output_dir / output_name

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for frame_id in range(start, valid_end_exclusive):
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_id} from {input_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = frame_rgb[y0 : y0 + crop_height, x0 : x0 + crop_width]
            frame_rgb = cv2.resize(
                frame_rgb,
                (target_width, target_height),
                interpolation=cv2.INTER_LANCZOS4,
            )
            frames.append(frame_rgb.astype(np.uint8))
        if pad_right:
            frames.extend(frames[-1].copy() for _ in range(pad_right))

        save_video(output_path, frames, fps=fps, quality=quality)
        clip_rows.append({
            "clip_index": clip_idx,
            "start_frame": start,
            "end_frame": end,
            "num_frames": clip_frames,
            "output_path": str(output_path),
            "valid_num_frames": valid_num_frames,
            "pad_right": pad_right,
            "padded_end_frame": padded_end,
        })
        padding_note = f", pad_right={pad_right}" if pad_right else ""
        print(
            f"[{clip_idx:03d}] source frames {start}..{end} "
            f"({valid_num_frames} valid{padding_note}) -> {output_path}"
        )

    cap.release()
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total_frames": total_frames,
        "fps": fps,
        "source_width": src_width,
        "source_height": src_height,
        "crop_x": x0,
        "crop_y": y0,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "target_width": target_width,
        "target_height": target_height,
        "clip_frames": clip_frames,
        "num_clips": len(clip_rows),
        "clips": clip_rows,
    }


def write_manifest(output_dir: Path, source_stem: str, manifest: Dict) -> None:
    json_path = output_dir / f"{source_stem}_splits_manifest.json"
    csv_path = output_dir / f"{source_stem}_splits_manifest.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_index",
                "start_frame",
                "end_frame",
                "num_frames",
                "output_path",
                "valid_num_frames",
                "pad_right",
                "padded_end_frame",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest["clips"])

    print(f"Manifest JSON: {json_path}")
    print(f"Manifest CSV:  {csv_path}")


def main(args):
    repo_root = Path(__file__).resolve().parents[2]
    validate_resolution_dimensions(
        args.resolution,
        height=args.height,
        width=args.width,
    )
    input_path = resolve_project_path(args.input, repo_root)
    output_dir = resolve_project_path(args.output_dir, repo_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    end_exclusive = total_frames
    if args.max_frames is not None and args.max_frames > 0:
        end_exclusive = min(end_exclusive, args.start_frame + args.max_frames)

    clip_starts = build_clip_starts(
        start_frame=args.start_frame,
        end_exclusive=end_exclusive,
        clip_frames=args.clip_frames,
        overlap=args.overlap,
        include_tail=args.include_tail,
        temporal_alignment=args.temporal_alignment,
    )
    manifest = read_and_write_clips(
        input_path=input_path,
        output_dir=output_dir,
        clip_starts=clip_starts,
        clip_frames=args.clip_frames,
        source_end_exclusive=end_exclusive,
        target_width=args.width,
        target_height=args.height,
        quality=args.quality,
        source_stem=args.source_stem,
    )
    manifest.update({
        "manifest_version": 2,
        "resolution": args.resolution,
        "start_frame": args.start_frame,
        "end_exclusive": end_exclusive,
        "overlap": args.overlap,
        "stride": args.clip_frames - args.overlap,
        "include_tail": args.include_tail,
        "tail_mode": "regular_stride_edge_padding" if args.include_tail else "drop",
        "padding_mode": "edge",
        "temporal_alignment": args.temporal_alignment,
        "quality": args.quality,
    })
    if args.manifest_metadata is not None:
        metadata_path = resolve_project_path(args.manifest_metadata, repo_root)
        with metadata_path.open("r", encoding="utf-8") as file:
            extra_metadata = json.load(file)
        if not isinstance(extra_metadata, dict):
            raise ValueError(
                f"--manifest_metadata must contain a JSON object: {metadata_path}"
            )
        conflicts = sorted(set(manifest).intersection(extra_metadata))
        if conflicts:
            raise ValueError(
                "Manifest metadata cannot replace splitter-owned fields: "
                f"{conflicts}"
            )
        manifest.update(extra_metadata)
    write_manifest(output_dir, args.source_stem or input_path.stem, manifest)

    print(
        "Summary: "
        f"{manifest['num_clips']} clips, "
        f"{args.clip_frames} frames each, overlap={args.overlap}, "
        f"stride={args.clip_frames - args.overlap}, "
        f"temporal_alignment={args.temporal_alignment}"
    )
    print(
        "Crop: "
        f"{manifest['source_width']}x{manifest['source_height']} "
        f"-> x={manifest['crop_x']} y={manifest['crop_y']} "
        f"w={manifest['crop_width']} h={manifest['crop_height']} "
        f"-> {manifest['target_width']}x{manifest['target_height']}"
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Split a video into overlapping fixed-length clips for Vista4D, "
            "with center crop and resize."
        )
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output_dir", default="./media/splits")
    parser.add_argument(
        "--source_stem",
        default=None,
        help=(
            "Optional output filename/manifest stem. The decoded input path is "
            "still recorded unchanged in the manifest."
        ),
    )
    parser.add_argument(
        "--manifest_metadata",
        default=None,
        help=(
            "Optional JSON object merged into the manifest. Existing "
            "splitter-owned keys may not be replaced."
        ),
    )
    parser.add_argument("--clip_frames", type=int, default=49)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument(
        "--temporal_alignment",
        type=int,
        default=4,
        help=(
            "Require every regular window start and stride to align to this "
            "many source frames. Wan/Vista4D uses 4."
        ),
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional number of source frames to consider from start_frame. Default uses the full video.",
    )
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--resolution",
        choices=("384p", "720p"),
        default="720p",
        help="Vista4D resolution profile; dimensions must match the profile.",
    )
    parser.add_argument("--quality", type=int, default=9)
    parser.add_argument(
        "--no_include_tail",
        dest="include_tail",
        action="store_false",
        help=(
            "Drop a remaining source tail instead of adding the next regular-stride "
            "clip and padding it to clip_frames."
        ),
    )
    parser.set_defaults(include_tail=True)
    main(parser.parse_args())
