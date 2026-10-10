"""Train the source-only predictor on synchronized multiview feature caches."""
from argparse import ArgumentParser
import json
from pathlib import Path

import torch

from jepa.features import FeatureCache
from jepa.predictor import PredictorConfig, TargetFeaturePredictor, feature_regression_loss, prepare_geometry


def load_pair(row, root):
    source, target = (FeatureCache.load(root / row[k]) for k in ("source", "target"))
    if source.role != "source" or target.role != "oracle":
        raise ValueError("Training pairs require source input and oracle supervision")
    if (source.encoder_id != target.encoder_id or source.features.shape != target.features.shape
            or source.num_frames != target.num_frames or tuple(source.image_size) != tuple(target.image_size)
            or not torch.equal(source.frame_positions, target.frame_positions)):
        raise ValueError("Training pairs must use synchronized grids and the same frozen teacher")
    geometry = torch.load(root / row["geometry"], map_location="cpu", weights_only=True)
    return source, target, prepare_geometry(source, geometry)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL: source, target, geometry paths relative to manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--memory-tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--occlusion-weight", type=float, default=1.0, help="Extra weight on source-rendered holes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or args.epochs < 1 or args.lr <= 0 or args.occlusion_weight < 0:
        raise ValueError("Need nonempty pairs, positive epochs/lr and nonnegative occlusion weight")
    torch.manual_seed(args.seed)
    source, _, _ = load_pair(rows[0], args.manifest.parent)
    encoder_id = source.encoder_id
    model = TargetFeaturePredictor(PredictorConfig(source.features.shape[1], args.hidden_dim, args.heads, args.memory_tokens, args.depth)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        total = 0.
        for index in torch.randperm(len(rows)).tolist():
            source, target, geometry = load_pair(rows[index], args.manifest.parent)
            if source.encoder_id != encoder_id:
                raise ValueError("All pairs must use the same teacher/preprocessing")
            inputs = {k: v.to(args.device) for k, v in geometry.items()}
            prediction = model(source.features.to(args.device), **inputs)
            weights = 1 + args.occlusion_weight * (1 - inputs["query_visibility"])
            loss = feature_regression_loss(prediction, target.features.to(args.device), weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite predictor loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item()
        model.save(args.output, encoder_id)
        print(json.dumps({"epoch": epoch + 1, "feature_loss": total / len(rows)}), flush=True)


if __name__ == "__main__":
    main()
