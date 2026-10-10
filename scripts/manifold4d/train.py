"""Fine-tune the Manifold4D trainable subset from supervised VAE/T5 caches."""
from argparse import ArgumentParser
import json
from pathlib import Path

import torch

from diffsynth.pipelines.manifold4d_training import manifold_training_loss, validate_training_example
from scripts.inference.inference_manifold4d import add_pipeline_args, get_pipeline, load_config


def main():
    parser = ArgumentParser(description=__doc__)
    add_pipeline_args(parser)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL containing a cache path per row")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--time_shift", type=float, default=1.0)
    parser.add_argument("--drop_probability", type=float, default=0.1)
    parser.add_argument("--unconditional_probability", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.vram_preset != "full":
        parser.error("Training requires --vram_preset full; inference offload is not a training strategy")
    if args.epochs < 1 or args.lr <= 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("epochs, lr and max_steps must be positive")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows or any("cache" not in row for row in rows):
        parser.error("Every training row needs a cache path")
    for row in rows:
        if not (args.manifest.parent / row["cache"]).is_file():
            raise FileNotFoundError(args.manifest.parent / row["cache"])
    torch.manual_seed(args.seed)
    # None checkpoint starts from Wan; a dedicated checkpoint initializes a
    # new fine-tuning run. This command does not claim optimizer-state resume.
    pipe = get_pipeline(args)
    model = pipe.dit.train()
    parameters = model.configure_trainable()
    # Keep the frozen VAE/T5 out of training VRAM; caches already contain them.
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    steps = 0
    for epoch in range(args.epochs):
        for index in torch.randperm(len(rows)).tolist():
            cache = args.manifest.parent / rows[index]["cache"]
            example = validate_training_example(torch.load(cache, map_location="cpu", weights_only=True))
            example = {k: v.to(device=pipe.device) if isinstance(v, torch.Tensor) else v for k, v in example.items()}
            loss = manifold_training_loss(model, example, prior_sigma=pipe.prior_sigma,
                                          time_shift=args.time_shift, drop_probability=args.drop_probability,
                                          unconditional_probability=args.unconditional_probability)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite Manifold4D flow loss for {cache}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            steps += 1
            print(json.dumps({"step": steps, "epoch": epoch + 1, "flow_loss": loss.item()}), flush=True)
            if args.max_steps is not None and steps >= args.max_steps:
                break
        model.save_checkpoint(args.output)
        if args.max_steps is not None and steps >= args.max_steps:
            break
    config = load_config(args)
    config["training"] = {"steps": steps, "lr": args.lr, "time_shift": args.time_shift,
                          "manifest": str(args.manifest.resolve()), "seed": args.seed,
                          "drop_probability": args.drop_probability, "unconditional_probability": args.unconditional_probability}
    import yaml
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
