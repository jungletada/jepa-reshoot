"""Manifold4D baseline using the existing Vista4D preprocessing file layout."""
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline
from ..models.wan_video_manifold4d import Manifold4DModel, camera_tokens, unwrap
from ..models.wan_video_text_encoder import HuggingfaceTokenizer
from ..utils.vista4d.media import apply_num_frames, crop_and_resize_pil, crop_and_resize_tensor
from .manifold4d import pool_video_mask, sample_manifold


def resize_camera_intrinsics(intrinsics, input_size, output_size):
    """Use exactly the integer centre crop of crop_and_resize_pil."""
    ih, iw = input_size
    oh, ow = output_size
    intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    if intrinsics.shape[-1] != 4:
        raise ValueError("Intrinsics must be [fx, fy, cx, cy] in pixel units")
    if input_size == output_size:
        return intrinsics
    if ih / iw > oh / ow:
        ch, cw = int(iw * oh / ow), iw
        x0, y0 = 0, (ih - ch) // 2
    else:
        ch, cw = ih, int(ih * ow / oh)
        x0, y0 = (iw - cw) // 2, 0
    intrinsics[..., 2:] -= [x0, y0]
    intrinsics *= [ow / cw, oh / ch, ow / cw, oh / ch]
    return intrinsics


class Manifold4DPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16,
                 positional_embedding_offset=31, prior_sigma=0.3, cond_stream_t="honest"):
        super().__init__(device, torch_dtype, height_division_factor=16, width_division_factor=16,
                         time_division_factor=4, time_division_remainder=1)
        self.dit = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.positional_embedding_offset = positional_embedding_offset
        self.prior_sigma = prior_sigma
        self.cond_stream_t = cond_stream_t
        self.checkpoint_loaded = False

    @classmethod
    def from_pretrained(cls, wan_checkpoint, manifold4d_checkpoint=None, *, device="cuda",
                        torch_dtype=torch.bfloat16, cpu_offload=False, vram_limit=None,
                        positional_embedding_offset=31, prior_sigma=0.3, cond_stream_t="honest",
                        encode_only=False):
        pipe = cls(device, torch_dtype, positional_embedding_offset, prior_sigma, cond_stream_t)
        root = Path(wan_checkpoint)
        patterns = ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"]
        if not encode_only:
            patterns.insert(0, "diffusion_pytorch_model*.safetensors")
        configs = []
        for pattern in patterns:
            files = sorted(str(p) for p in root.glob(pattern))
            if not files:
                raise FileNotFoundError(f"Wan2.1 base model files not found: {root / pattern}")
            configs.append(ModelConfig(path=files, skip_download=True,
                                       offload_device="cpu" if cpu_offload else None,
                                       offload_dtype=torch_dtype if cpu_offload else None))
        tokenizer = root / "google/umt5-xxl"
        if not tokenizer.is_dir():
            raise FileNotFoundError(f"Wan tokenizer directory not found: {tokenizer}")
        if not encode_only and manifold4d_checkpoint is not None:
            # Fail before loading 14B if the dedicated weights are incomplete.
            for name in ("conditioning_modules.pt", "camera_encoder.pt", "self_attn_full.pt"):
                if not (Path(manifold4d_checkpoint) / name).is_file():
                    raise FileNotFoundError(f"Manifold4D checkpoint requires {Path(manifold4d_checkpoint) / name}")
        pool = pipe.download_and_load_models(configs, vram_limit=vram_limit)
        pipe.vae = pool.fetch_model("wan_video_vae")
        pipe.text_encoder = pool.fetch_model("wan_video_text_encoder")
        if pipe.vae is None or pipe.text_encoder is None or pipe.vae.upsampling_factor != 8:
            raise ValueError("Require the native Wan2.1 16-channel VAE and UMT5 encoder")
        pipe.tokenizer = HuggingfaceTokenizer(str(tokenizer), seq_len=512, clean="whitespace")
        if not encode_only:
            base = pool.fetch_model("wan_video_dit")
            if base is None or base.dim != 5120 or len(base.blocks) != 40 or base.in_dim != 16 or base.has_image_input:
                raise ValueError("The released baseline requires Wan2.1-T2V-14B (16 channels, 40 blocks)")
            pipe.dit = Manifold4DModel(base, positional_embedding_offset, cond_stream_t)
            if manifold4d_checkpoint is not None:
                pipe.dit.load_checkpoint(manifold4d_checkpoint)
                pipe.checkpoint_loaded = True
            # Preserve offload wrappers' device policies. The four new convs
            # are small; large projectors follow the base attention's policy.
            for name in pipe.dit.stream_names:
                getattr(pipe.dit, name).to(device=device, dtype=torch_dtype)
            if not cpu_offload:
                for block in base.blocks:
                    unwrap(block).cam_encoder.to(device=device, dtype=torch_dtype)
                    unwrap(block).projector.to(device=device, dtype=torch_dtype)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe.eval()

    @torch.no_grad()
    def encode_prompt(self, prompt):
        self.load_models_to_device(["text_encoder"])
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(self.device), mask.to(self.device)
        embedding = self.text_encoder(ids, mask)
        for i, length in enumerate(mask.gt(0).sum(1).tolist()):
            embedding[i, length:] = 0
        return embedding.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def encode_video(self, video, height, width, num_frames, tiled=False):
        video = apply_num_frames(video, num_frames)
        video = crop_and_resize_pil(video, height, width)
        self.load_models_to_device(["vae"])
        return self.vae.encode(self.preprocess_video(video), device=self.device, tiled=tiled).to(self.device, self.torch_dtype)

    @torch.no_grad()
    def prepare_inputs(self, *, source_video, point_cloud_video, point_cloud_alpha_mask,
                       source_cam_c2w, source_intrinsics, target_cam_c2w, target_intrinsics,
                       source_alpha_mask=None, source_motion_mask=None, point_cloud_motion_mask=None,
                       height=384, width=672, num_frames=49, tiled=False):
        if height < 16 or width < 16 or height % 16 or width % 16 or num_frames < 1 or (num_frames - 1) % 4:
            raise ValueError("Require positive 16-aligned spatial dimensions and 4n+1 frames")
        lengths = [len(v) for v in (source_video, point_cloud_video, point_cloud_alpha_mask,
                                    source_cam_c2w, source_intrinsics, target_cam_c2w, target_intrinsics)]
        if len(set(lengths)) != 1 or lengths[0] < num_frames:
            raise ValueError("Source/render videos, coverage and cameras must have matching lengths >= num_frames")
        src_size = (source_video[0].height, source_video[0].width)
        render_size = (point_cloud_video[0].height, point_cloud_video[0].width)

        def masks(alpha, motion, video_size):
            if alpha is None:
                alpha = np.ones((lengths[0], *video_size), dtype=np.float32)
            if motion is None:
                motion = np.zeros_like(alpha)
            if np.shape(alpha) != (lengths[0], *video_size) or np.shape(motion) != np.shape(alpha):
                raise ValueError("Pixel masks must match the corresponding video's [T, H, W] shape")
            value = np.stack((apply_num_frames(alpha, num_frames), apply_num_frames(motion, num_frames)), 0)
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)[None]
            value = crop_and_resize_tensor(value, height, width, mode="bilinear")
            return pool_video_mask(value)

        source_mask = masks(source_alpha_mask, source_motion_mask, src_size)
        render_mask = masks(point_cloud_alpha_mask, point_cloud_motion_mask, render_size)
        source = self.encode_video(source_video, height, width, num_frames, tiled)
        render = self.encode_video(point_cloud_video, height, width, num_frames, tiled)
        if source.shape != render.shape or source.shape[2:] != source_mask.shape[2:]:
            raise ValueError("VAE outputs and pooled masks do not share the latent grid")
        grid = (source.shape[2], source.shape[3] // 2, source.shape[4] // 2)

        def camera(c2w, intrinsics, size):
            c2w = torch.as_tensor(apply_num_frames(c2w, num_frames).copy(), device=self.device, dtype=torch.float32)[None]
            intrinsics = resize_camera_intrinsics(apply_num_frames(intrinsics, num_frames), size, (height, width))
            intrinsics = torch.as_tensor(intrinsics, device=self.device, dtype=torch.float32)[None]
            return camera_tokens(c2w, intrinsics, grid, (height, width))

        rays = torch.cat((camera(target_cam_c2w, target_intrinsics, render_size),
                          camera(source_cam_c2w, source_intrinsics, src_size)), 1)
        return {"render_latents": render, "source_latents": source,
                "render_mask": render_mask, "source_mask": source_mask, "camera_embedding": rays}

    @torch.no_grad()
    def __call__(self, *, prompt, negative_prompt="", seed=(42,), cfg_scale=5.0,
                 cfg_merge=False, num_inference_steps=50, sigma_shift=5.0,
                 solver="unipc", rand_device=None, output_type="quantized",
                 progress_bar_cmd=tqdm, **inputs):
        if self.dit is None or not self.checkpoint_loaded:
            raise ValueError("Inference requires trained Manifold4D weights; Vista4D/Wan weights alone are insufficient")
        if not seed:
            raise ValueError("At least one seed is required")
        rand_device = self.device if rand_device is None else rand_device
        prepared = self.prepare_inputs(**inputs)
        prepared = {k: v.repeat(len(seed), *([1] * (v.ndim - 1))) for k, v in prepared.items()}
        render = prepared.pop("render_latents")
        context = self.encode_prompt([prompt] * len(seed))
        negative = self.encode_prompt([negative_prompt] * len(seed)) if cfg_scale != 1 else None
        noise = torch.cat([self.generate_noise((1, *render.shape[1:]), s, rand_device,
                                               torch_dtype=torch.float32) for s in seed])
        self.load_models_to_device(["dit"])
        try:
            latent = sample_manifold(self.dit, render, prepared["render_mask"][:, :1], noise,
                                     context, prepared, negative_context=negative,
                                     prior_sigma=self.prior_sigma, cfg_scale=cfg_scale,
                                     num_steps=num_inference_steps, shift=sigma_shift, solver=solver,
                                     cfg_merge=cfg_merge, progress=progress_bar_cmd)
            self.load_models_to_device(["vae"])
            video = self.vae.decode(latent.to(self.torch_dtype), device=self.device, tiled=inputs.get("tiled", False))
            if output_type == "floatpoint":
                return video
            if output_type != "quantized":
                raise ValueError("output_type must be quantized or floatpoint")
            return [[self.vae_output_to_image(frame, pattern="C H W") for frame in clip.unbind(1)] for clip in video]
        finally:
            self.load_models_to_device([])
