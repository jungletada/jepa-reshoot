"""Freeze Vista4D and fit JEPA cross-attention with target-video flow matching."""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

import torch
import yaml

from jepa.adapter import AdapterConfig, JEPAAdapter
from jepa.features import FeatureCache
from jepa.training import adapter_flow_loss, prepare_adapter_example


def main():
    from scripts.inference.inference import add_model_args, get_pipeline
    parser = ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL: input_folder, target_video, prompt, features (list of caches)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-adapter", help="Optional Oracle-trained adapter for subsequent predicted/mixed-condition training")
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 19, 29, 39])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--spatial-pool", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--tile_vae", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.use_usp or args.vram_preset != "full" or args.fp8_compute:
        raise ValueError("Adapter training currently requires --vram_preset full, no FP8 and no USP")
    if args.epochs < 1 or args.lr <= 0 or (args.num_frames - 1) % 4:
        raise ValueError("Require positive epochs/lr and a 4n+1 frame clip")
    root = args.manifest.parent
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("features"), list) or not row["features"] for row in rows):
        raise ValueError("Each training row requires a nonempty list of feature caches")
    torch.manual_seed(args.seed)
    first = FeatureCache.load(root / rows[0]["features"][0])
    config = yaml.safe_load(Path(args.vista4d_config_path).read_text())
    args.jepa_adapter = args.init_adapter
    pipe = get_pipeline(args, config).eval().requires_grad_(False)
    if pipe.dit2 is not None:
        raise ValueError("Only the single-DiT Vista4D model is supported")
    if args.init_adapter:
        if pipe.jepa_encoder_id != first.encoder_id:
            raise ValueError("Initial adapter and caches must share the teacher")
    else:
        adapter_config = AdapterConfig(first.features.shape[1], args.hidden_dim, args.heads, tuple(args.layers), args.spatial_pool)
        pipe.dit.jepa_adapter = JEPAAdapter(pipe.dit.dim, len(pipe.dit.blocks), adapter_config)
    adapter = pipe.dit.jepa_adapter.to(device=pipe.device, dtype=torch.float32).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    pipe.scheduler.set_timesteps(1000, training=True, shift=args.sigma_shift)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        total = 0.
        for index in torch.randperm(len(rows)).tolist():
            row = rows[index]
            # Uniform condition mixing trains the same generator on Oracle and/or
            # source-only predictions. The target RGB is used solely for the loss.
            feature_path = row["features"][torch.randint(len(row["features"]), ()).item()]
            cache = FeatureCache.load(root / feature_path)
            features = cache.for_generation(mode=cache.role, num_frames=args.num_frames,
                image_size=(args.height, args.width), batch_size=1, encoder_id=first.encoder_id).to(pipe.device)
            example = Namespace(**{**vars(args), "input_folder": str(root / row["input_folder"]),
                "prompt": row["prompt"], "negative_prompt": "", "seed": [args.seed + index]})
            inputs, target = prepare_adapter_example(pipe, example, config, root / row["target_video"])
            loss = adapter_flow_loss(pipe, inputs, target, features)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite adapter loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
            optimizer.step()
            total += loss.item()
        adapter.save(args.output, first.encoder_id)
        print(json.dumps({"epoch": epoch + 1, "flow_loss": total / len(rows)}), flush=True)


if __name__ == "__main__":
    main()
