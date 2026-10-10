"""Supervised Manifold4D flow training from frozen VAE/T5 caches."""
import math

import torch

from .manifold4d import balanced_flow_loss, flow_path, manifold_prior


def manifold_training_loss(model, example, *, prior_sigma=0.3, time=None, time_shift=1.0,
                           noise=None, drop_probability=0.1, unconditional_probability=0.05,
                           dynamic_weight=2.0, use_gradient_checkpointing=True, generator=None):
    """Train the shifted endpoint, not Gaussian-to-target flow on Vista4D weights.

    Each stream drops independently; render dropout zeros coverage, source
    dropout uses Gaussian noise with an honest t=1000, and prompt dropout
    uses the encoded empty prompt. Target latents are used only for the loss.
    Render corruption and temporal reversal belong to pair preparation.
    """
    if not (0 <= drop_probability <= 1 and 0 <= unconditional_probability <= 1):
        raise ValueError("Drop probabilities must lie in [0, 1]")
    if not math.isfinite(time_shift) or time_shift <= 0:
        raise ValueError("time_shift must be finite and positive")
    target, render = example["target_latents"], example["render_latents"]
    b = target.shape[0]
    device = target.device
    if noise is None:
        noise = torch.randn(target.shape, device=device, generator=generator, dtype=torch.float32)
    if time is None:
        time = torch.rand(b, device=device, generator=generator)
        time = time_shift * time / (1 + (time_shift - 1) * time)
    else:
        time = torch.as_tensor(time, device=device, dtype=torch.float32)
        if time.ndim == 0:
            time = time.expand(b)
    fully_unconditional = torch.rand(b, device=device, generator=generator) < unconditional_probability
    drops = [(torch.rand(b, device=device, generator=generator) < drop_probability) | fully_unconditional
             for _ in range(4)]
    drop_source, drop_render, drop_camera, drop_prompt = drops
    keep_render = (~drop_render)[:, None, None, None, None]
    render_mask = example["render_mask"] * keep_render
    alpha = render_mask[:, :1]
    prior = manifold_prior(render, alpha, noise, prior_sigma)
    xt, velocity = flow_path(target, prior, time)
    source = example["source_latents"]
    source_noise = torch.randn(source.shape, device=device, generator=generator, dtype=torch.float32).to(source.dtype)
    source = torch.where(drop_source[:, None, None, None, None], source_noise, source)
    source_mask = example["source_mask"] * (~drop_source)[:, None, None, None, None]
    rays = example["camera_embedding"] * (~drop_camera)[:, None, None]
    context = torch.where(drop_prompt[:, None, None], example["empty_context"], example["context"])
    prediction = model(xt, time * 1000, context, source_latents=source,
                       render_mask=render_mask, source_mask=source_mask, camera_embedding=rays,
                       source_timestep=drop_source.float() * 1000,
                       use_gradient_checkpointing=use_gradient_checkpointing)
    return balanced_flow_loss(prediction, velocity, alpha,
                              example.get("target_motion_mask"), dynamic_weight)


def validate_training_example(example):
    required = {"source_latents", "render_latents", "target_latents", "source_mask",
                "render_mask", "camera_embedding", "context", "empty_context"}
    if not isinstance(example, dict) or required - example.keys():
        raise ValueError(f"Training cache requires {sorted(required)}")
    if any(not isinstance(example[k], torch.Tensor) or not torch.isfinite(example[k]).all() for k in required):
        raise ValueError("Training cache must contain finite tensors")
    target = example["target_latents"]
    if target.ndim != 5 or target.shape[0] != 1 or target.shape[1] != 16:
        raise ValueError("Training caches require [1, 16, T, H, W] Wan2.1 latents")
    if any(example[k].shape != target.shape for k in ("source_latents", "render_latents")):
        raise ValueError("Source/render/target caches must share the latent grid")
    b, _, t, h, w = target.shape
    if h % 2 or w % 2:
        raise ValueError("Cached latent height/width must be even")
    for key in ("source_mask", "render_mask"):
        mask = example[key]
        if mask.shape != (b, 2, t, h, w) or (mask < 0).any() or (mask > 1).any():
            raise ValueError(f"{key} must be a 2-channel coverage/motion mask in [0,1]")
    if example["camera_embedding"].shape != (b, 2 * t * (h // 2) * (w // 2), 6):
        raise ValueError("Cached camera rays must match the two-stream token layout")
    if example["context"].ndim != 3 or example["context"].shape != (b, 512, 4096) or example["empty_context"].shape != example["context"].shape:
        raise ValueError("Cache text must use Wan UMT5 [1, 512, 4096] embeddings")
    if "target_motion_mask" in example:
        motion = example["target_motion_mask"]
        if not isinstance(motion, torch.Tensor) or motion.shape != (b, 1, t, h, w) or not torch.isfinite(motion).all() or (motion < 0).any() or (motion > 1).any():
            raise ValueError("Target motion cache must be [1,1,T,H,W] in [0,1]")
    return example
