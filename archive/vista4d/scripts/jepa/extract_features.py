"""Encode a complete prepared clip with a frozen, locally installed JEPA teacher."""
from argparse import ArgumentParser

import imageio.v3 as iio
import numpy as np
import torch

from jepa.teacher import load_local_teacher


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--role", required=True, choices=["source", "projected", "oracle"])
    parser.add_argument("--teacher-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", default="vjepa2_1_vit_large_384")
    parser.add_argument("--checkpoint-key", default="ema_encoder")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 384], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    video = np.stack(list(iio.imiter(args.video)))
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("Expected an RGB video")
    video = torch.from_numpy(video.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().div(255).to(args.device)
    teacher = load_local_teacher(args.teacher_repo, args.checkpoint, args.architecture, args.checkpoint_key, tuple(args.size)).to(args.device)
    cache = teacher.encode(video, args.role)
    cache.save(args.output)
    print(f"Saved {args.role} features {tuple(cache.features.shape)} to {args.output}")


if __name__ == "__main__":
    main()
