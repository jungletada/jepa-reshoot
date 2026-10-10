"""Shared video identity, full-timeline geometry and experiment paths (no torch)."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = {"384p": (672, 384), "720p": (1280, 720)}


def probe_source(video):
    video = Path(video).resolve()
    result = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,duration",
        "-of", "json", str(video),
    ], text=True)
    stream = json.loads(result)["streams"][0]
    frames = int(stream["nb_read_frames"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    if frames <= 0 or fps <= 0:
        raise ValueError("Video must have positive decoded frame count and FPS")
    return dict(path=str(video), frames=frames, fps=fps, width=int(stream["width"]),
                height=int(stream["height"]), avg_frame_rate=stream["avg_frame_rate"],
                r_frame_rate=stream["r_frame_rate"])


def split_directory(resolution, baseline=False):
    folder = ROOT / ("media/splits" if baseline else "media/flowlong_splits")
    return folder if resolution == "384p" else folder / resolution


def window_layout(frames, stride=24):
    if frames <= 0:
        raise ValueError("No input frames")
    starts = [0]
    if frames > 49:
        starts = [stride * i for i in range(1 + (frames - 49 + stride - 1) // stride)]
    return dict(frames=frames, windows=len(starts), starts=starts,
                tail_valid=frames - starts[-1], tail_padding=starts[-1] + 49 - frames,
                flowlong_supported=len(starts) >= 2)

