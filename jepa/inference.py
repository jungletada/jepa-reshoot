"""Shared CLI/cache boundary for single-clip and FlowLong inference."""
import math

import torch

from .features import FeatureCache, ROLES


def add_jepa_args(parser):
    parser.add_argument("--jepa_mode", choices=["none", *sorted(ROLES)], default="none")
    parser.add_argument("--jepa_features", nargs="+", help="One clip cache, or one cache per FlowLong window in manifest order")
    parser.add_argument("--jepa_adapter", help="Trained JEPA adapter checkpoint")
    parser.add_argument("--jepa_scale", type=float, default=1.0)


def validate_jepa_args(args):
    mode = getattr(args, "jepa_mode", "none")
    features = getattr(args, "jepa_features", None)
    if mode == "none" and features:
        raise ValueError("Select an explicit JEPA mode when supplying a feature cache")
    if mode != "none" and (not features or not getattr(args, "jepa_adapter", None)):
        raise ValueError("JEPA modes require both --jepa_features and a trained --jepa_adapter")
    if not math.isfinite(getattr(args, "jepa_scale", 1.0)):
        raise ValueError("JEPA scale must be finite")


def load_jepa_inputs(args, pipe, batch_size, *, windowed=False):
    validate_jepa_args(args)
    if getattr(args, "jepa_mode", "none") == "none":
        return {}
    paths = args.jepa_features
    expected = batch_size if windowed else 1
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} JEPA feature cache(s), got {len(paths)}")
    features = [FeatureCache.load(path).for_generation(
        mode=args.jepa_mode, num_frames=args.num_frames, image_size=(args.height, args.width),
        batch_size=1 if windowed else batch_size, encoder_id=pipe.jepa_encoder_id,
    ) for path in paths]
    return {"jepa_features": torch.cat(features, dim=0), "jepa_scale": args.jepa_scale}
