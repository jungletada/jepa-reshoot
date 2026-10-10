"""Adapter-only training using Vista4D's native flow-matching schedule."""
import torch
from torch.nn import functional as F


def adapter_flow_loss(pipe, inputs, target_latents, features):
    index = torch.randint(len(pipe.scheduler.timesteps), (1,))
    timestep = pipe.scheduler.timesteps[index].to(device=target_latents.device, dtype=target_latents.dtype)
    noise = torch.randn_like(target_latents)
    latents = pipe.scheduler.add_noise(target_latents, noise, timestep)
    target = pipe.scheduler.training_target(target_latents, noise, timestep)
    prediction = pipe.model_fn(dit=pipe.dit, **inputs, latents=latents, timestep=timestep,
                              jepa_features=features, use_gradient_checkpointing=True)
    return F.mse_loss(prediction.float(), target.float()) * pipe.scheduler.training_weight(timestep)


MODEL_INPUTS = {"source_video_latents", "point_cloud_video_latents", "source_mask_latents",
                "point_cloud_mask_latents", "cam_emb", "context", "clip_feature", "y", "y_empty"}


@torch.no_grad()
def prepare_adapter_example(pipe, args, vista_config, target_video):
    from scripts.inference.inference import get_inputs
    from utils.media import load_video, np_to_pil
    inputs, source_fps = get_inputs(args, vista_config)
    if len(inputs["source_video"][0]) != args.num_frames or len(inputs["point_cloud_video"][0]) != args.num_frames:
        raise ValueError("Training requires already prepared clips with the exact frame count")
    for videos in (inputs["source_video"], inputs["point_cloud_video"]):
        if any(frame.size != (args.width, args.height) for video in videos for frame in video):
            raise ValueError("Training clips must already have the configured resolution/FOV")
    shared, positive, _ = pipe._prepare_inference_inputs(**{
        "input_image": None, "end_image": None, **inputs,
        "seed": args.seed, "rand_device": "cpu", "batch_size": 1,
        "cfg_scale": 1., "cfg_merge": False, "sigma_shift": args.sigma_shift,
        "num_inference_steps": 1000, "tiled": args.tile_vae,
        "tile_size": (30, 52), "tile_stride": (15, 26),
    })
    target, fps = load_video(str(target_video))
    if target.shape != (args.num_frames, args.height, args.width, 3) or abs(fps - source_fps) > .01:
        raise ValueError("Target video must be synchronized and match the prepared source clip dimensions/FPS")
    pipe.load_models_to_device(["vae"])
    target = pipe.preprocess_video(np_to_pil(target))
    target = pipe.vae.encode(target, device=pipe.device, tiled=args.tile_vae).to(device=pipe.device, dtype=pipe.torch_dtype)
    model_inputs = {k: v for k, v in {**shared, **positive}.items() if k in MODEL_INPUTS}
    return model_inputs, target
