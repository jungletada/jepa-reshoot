from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import cv2
import imageio
import numpy as np


DEFAULT_INPUT = (
    "data/PRO_VID_20260602_153852_00_023/"
    "PRO_VID_20260602_153852_00_023_ud.mp4"
)
DEFAULT_OUTPUT_NAME = "PRO_VID_20260602_153852_00_023_frames170_218_720p49"


def resolve_project_path(value: str, repo_root: Path) -> Path:
    """Resolve paths from either Vista4D/ or the parent video-stablization root."""
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


def read_clip(
    input_path: Path,
    start_frame: int,
    num_frames: int,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, float, dict]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_frame < 0:
        raise ValueError(f"--start_frame must be >= 0, got {start_frame}")
    if total_frames > 0 and start_frame + num_frames > total_frames:
        raise ValueError(
            f"Need frames {start_frame}..{start_frame + num_frames - 1}, "
            f"but input only has {total_frames} frames."
        )

    x0, y0, crop_width, crop_height = center_crop_box(
        src_width, src_height, target_width, target_height
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for frame_id in range(start_frame, start_frame + num_frames):
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
        frames.append(frame_rgb)

    cap.release()
    video = np.stack(frames, axis=0).astype(np.uint8)
    metadata = {
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
        "start_frame": start_frame,
        "end_frame": start_frame + num_frames - 1,
        "num_frames": num_frames,
    }
    return video, fps, metadata


def save_video(output_path: Path, video: np.ndarray, fps: float, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        quality=quality,
        macro_block_size=1,
    )
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()


def main(args):
    repo_root = Path(__file__).resolve().parents[2]
    input_path = resolve_project_path(args.input, repo_root)
    output_dir = resolve_project_path(args.output_dir, repo_root)
    output_name = args.output_name
    if output_name.endswith(".mp4"):
        output_name = output_name[:-4]
    output_path = output_dir / f"{output_name}.mp4"

    video, fps, metadata = read_clip(
        input_path=input_path,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        target_width=args.width,
        target_height=args.height,
    )
    save_video(output_path, video, fps=fps, quality=args.quality)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        "Frames: "
        f"{metadata['start_frame']}..{metadata['end_frame']} "
        f"({metadata['num_frames']} frames, 0-based)"
    )
    print(
        "Crop: "
        f"{metadata['source_width']}x{metadata['source_height']} "
        f"-> x={metadata['crop_x']} y={metadata['crop_y']} "
        f"w={metadata['crop_width']} h={metadata['crop_height']} "
        f"-> {metadata['target_width']}x{metadata['target_height']}"
    )
    print(f"FPS: {fps:.6g}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Prepare a custom 49-frame Vista4D single-video clip by center-cropping "
            "to the target aspect ratio and resizing."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default="./media/single")
    parser.add_argument("--output_name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--start_frame", type=int, default=170, help="0-based first frame index.")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=9)
    main(parser.parse_args())
