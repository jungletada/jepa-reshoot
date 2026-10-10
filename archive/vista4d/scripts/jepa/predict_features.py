"""Predict target-view JEPA features without reading target video or target depth."""
from argparse import ArgumentParser

import torch

from jepa.features import FeatureCache
from jepa.predictor import TargetFeaturePredictor, prepare_geometry


@torch.no_grad()
def predict(checkpoint, source_path, geometry_path, device):
    model, encoder_id = TargetFeaturePredictor.load(checkpoint)
    source = FeatureCache.load(source_path)
    if source.role != "source" or source.encoder_id != encoder_id:
        raise ValueError("Predictor requires SOURCE features from its training encoder")
    geometry = torch.load(geometry_path, map_location="cpu", weights_only=True)
    inputs = prepare_geometry(source, geometry)
    features = model.to(device).eval()(source.features.to(device), **{k: v.to(device) for k, v in inputs.items()})
    return FeatureCache(features.cpu(), source.frame_positions, source.num_frames, source.image_size, encoder_id, "predicted")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    predict(args.checkpoint, args.source, args.geometry, args.device).save(args.output)


if __name__ == "__main__":
    main()
