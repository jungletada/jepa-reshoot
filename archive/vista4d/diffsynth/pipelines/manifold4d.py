"""Manifold4D prior, mask layout and flow objective (paper Eqs. 4–5).

Reference: https://arxiv.org/abs/2608.28174 and the released implementation
https://github.com/ManifoldTechLtd/Manifold4D at 14e1989e0e854a1c05eae0cf09533a7558adf20b.
Tensors use [batch, channels, latent_frames, height, width].
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def _mask(mask, reference, name):
    if mask.shape != (reference.shape[0], 1, *reference.shape[2:]):
        raise ValueError(f"{name} must have shape [B, 1, T, H, W] matching the latent")
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return mask.to(device=reference.device, dtype=torch.float32)


def manifold_prior(render, coverage, noise, sigma=0.3):
    """One shared epsilon for both branches; retain fractional coverage."""
    if render.ndim != 5 or render.shape != noise.shape:
        raise ValueError("Render and noise must have the same [B, C, T, H, W] shape")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("prior sigma must be finite and nonnegative")
    alpha = _mask(coverage, render, "coverage")
    eps = noise.to(device=render.device, dtype=torch.float32)
    return alpha * (render.float() + sigma * eps) + (1 - alpha) * eps


def pool_video_mask(mask, stride=(4, 8, 8)):
    """Official avgpool layout: frame 0 alone, then nonoverlapping groups of 4.

    The released model pools to VAE resolution, then its 2-channel mask conv
    patchifies spatially. It does not use Vista4D's 512-channel allpack masks.
    """
    st, sh, sw = stride
    if mask.ndim != 5 or min(mask.shape) < 1:
        raise ValueError("Mask must be a nonempty [B, C, T, H, W] tensor")
    if (mask.shape[2] - 1) % st or mask.shape[3] % sh or mask.shape[4] % sw:
        raise ValueError("Mask requires 4n+1 frames and spatial dimensions divisible by the VAE stride")
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError("Mask must contain finite coverage in [0, 1]")
    first = F.avg_pool3d(mask[:, :, :1].float(), (1, sh, sw))
    if mask.shape[2] == 1:
        return first
    return torch.cat((first, F.avg_pool3d(mask[:, :, 1:].float(), stride)), dim=2)


def flow_path(target, prior, time):
    """t=0 clean, t=1 prior; the reverse ODE integrates v=prior-target."""
    if target.ndim != 5 or target.shape != prior.shape:
        raise ValueError("Target and prior must have matching [B, C, T, H, W] shapes")
    time = torch.as_tensor(time, device=target.device, dtype=torch.float32)
    if time.ndim == 0:
        time = time.expand(target.shape[0])
    if time.shape != (target.shape[0],) or not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
        raise ValueError("Flow time must be a scalar or [B] in [0, 1]")
    t = time[:, None, None, None, None]
    return (1 - t) * target.float() + t * prior.float(), prior.float() - target.float()


def balanced_flow_loss(prediction, velocity, coverage, motion=None, dynamic_weight=2.0):
    """Appendix B: equal covered/hole normalization, double dynamic weight.

    Fractional masks weight membership in each region. A missing region is
    omitted, so all-covered/all-hole examples keep the same loss scale.
    """
    if prediction.shape != velocity.shape or prediction.ndim != 5:
        raise ValueError("Prediction and velocity must have matching [B, C, T, H, W] shapes")
    if not math.isfinite(dynamic_weight) or dynamic_weight < 1:
        raise ValueError("dynamic_weight must be finite and >= 1")
    alpha = _mask(coverage, velocity, "coverage")
    moving = torch.zeros_like(alpha) if motion is None else _mask(motion, velocity, "motion")
    error = (prediction.float() - velocity.float()).square().mean(dim=1, keepdim=True)
    error = error * (1 + (dynamic_weight - 1) * moving)
    dims = (1, 2, 3, 4)
    masses = torch.stack((alpha.sum(dims), (1 - alpha).sum(dims)), dim=1)
    sums = torch.stack(((error * alpha).sum(dims), (error * (1 - alpha)).sum(dims)), dim=1)
    valid = masses > 0
    regions = sums / masses.clamp_min(1e-12)
    return ((regions * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()


def make_scheduler(num_steps=50, shift=5.0, solver="unipc", device="cpu"):
    if num_steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Sampling requires positive steps and a finite positive shift")
    # Wan's FlowUniPC starts at float32(1 - 1/1000), rather than exactly 1.
    # Supply its grid explicitly: diffusers' default flow grid differs.
    sigmas = np.linspace(float(np.float32(0.999)), 0, num_steps + 1)[:-1]
    if solver == "unipc":
        from diffusers import UniPCMultistepScheduler
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True,
            flow_shift=shift, solver_order=2, solver_type="bh2",
            predict_x0=True, lower_order_final=True, final_sigmas_type="zero",
        )
        scheduler.set_timesteps(device=device, sigmas=sigmas)
        return scheduler
    if solver == "euler":
        from ..diffusion import FlowMatchScheduler
        scheduler = FlowMatchScheduler("Wan")
        scheduler.set_timesteps(num_steps, shift=shift)
        return scheduler
    raise ValueError(f"Unknown solver: {solver}")


@torch.no_grad()
def sample_manifold(model, render, coverage, noise, context, conditions, *,
                    negative_context=None, prior_sigma=0.3, cfg_scale=5.0,
                    num_steps=50, shift=5.0, solver="unipc", cfg_merge=False,
                    progress=lambda x: x):
    """Render RGB is consumed here once; it never enters model conditions."""
    if not math.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError("CFG scale must be finite and nonnegative")
    if {"render", "render_latents", "point_cloud_video_latents"}.intersection(conditions):
        raise ValueError("Manifold4D has no persistent render RGB condition")
    if cfg_scale != 1 and negative_context is None:
        raise ValueError("Text CFG requires the encoded negative prompt")
    if negative_context is not None and negative_context.shape != context.shape:
        raise ValueError("Positive/negative contexts must have matching shapes")
    latents = manifold_prior(render, coverage, noise, prior_sigma)
    scheduler = make_scheduler(num_steps, shift, solver, latents.device)
    merged_conditions = None
    if cfg_merge and cfg_scale != 1:
        merged_conditions = {k: torch.cat((v, v), 0) for k, v in conditions.items()}
    for step in progress(scheduler.timesteps):
        timestep = torch.as_tensor(step, device=latents.device, dtype=torch.float32).expand(latents.shape[0])
        if merged_conditions is not None:
            pos, neg = model(torch.cat((latents, latents), 0), timestep.repeat(2),
                             torch.cat((context, negative_context), 0), **merged_conditions).chunk(2)
        else:
            pos = model(latents, timestep, context, **conditions)
            neg = model(latents, timestep, negative_context, **conditions) if cfg_scale != 1 else pos
        velocity = neg + cfg_scale * (pos - neg)
        if solver == "unipc":
            latents = scheduler.step(velocity, step, latents, return_dict=False)[0]
        else:
            latents = scheduler.step(velocity, step, latents)
    return latents
