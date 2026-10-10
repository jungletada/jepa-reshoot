"""Released Manifold4D two-stream architecture on this repo's Wan backbone.

Adapted from ManifoldTechLtd/Manifold4D (Apache-2.0), commit
14e1989e0e854a1c05eae0cf09533a7558adf20b, model14b/manifold4d_14b.py.
The existing DiTBlock already implements input_x_add camera injection and
the post-self-attention projector. No Vista4D RGB-render stream is installed.
"""
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..core import AutoWrappedModule, AutoWrappedLinear
from ..utils.vista4d.camera import get_plucker_embedding
from .wan_video_dit import AttentionModule, flash_attention, sinusoidal_embedding_1d
from utils.vista4d_checkpoint import wrapped_key_map


def unwrap(module):
    while isinstance(module, AutoWrappedModule):
        module = module.module
    return module


class ManifoldAttention(AttentionModule):
    def forward(self, q, k, v):
        # Installed flash-attn has no CPU/fp32 kernel. Keep CPU tests and
        # fp32 diagnostic runs independent of CUDA extension availability.
        return flash_attention(q, k, v, self.num_heads,
                               compatibility_mode=not q.is_cuda or q.dtype == torch.float32)


class ManifoldLayerNorm(nn.LayerNorm):
    def forward(self, x):
        # Per-token modulation produces fp32 residuals. Wan's norm uses fp32
        # accumulation even when its stored affine weights are bf16.
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)


def camera_tokens(c2w, intrinsics, grid, image_size):
    """Half-pixel token centres and frame indices [0,4,8,...], as released.

    intrinsics are [fx, fy, cx, cy] in pixels at image_size. Calling the
    existing helper at the scaled token resolution avoids its extra half
    patch shift in the height_dit/width_dit branch.
    """
    frames, height, width = grid
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4) or intrinsics.shape != (*c2w.shape[:2], 4):
        raise ValueError("Cameras require [B, T, 4, 4] c2w and [B, T, 4] intrinsics")
    if c2w.shape[1] < 4 * (frames - 1) + 1:
        raise ValueError("Camera trajectory is shorter than the latent clip")
    if not torch.isfinite(c2w).all() or not torch.isfinite(intrinsics).all() or (intrinsics[..., :2] <= 0).any():
        raise ValueError("Cameras must be finite with positive focal lengths")
    indices = torch.arange(frames, device=c2w.device) * 4
    c2w = c2w[:, indices].float()
    intrinsics = intrinsics[:, indices].to(c2w)
    scale = intrinsics.new_tensor([width / image_size[1], height / image_size[0]] * 2)
    rays = get_plucker_embedding(intrinsics * scale, c2w, height, width)
    return rays.flatten(1, 3)


def two_stream_freqs(base, grid, offset, device):
    f, h, w = grid
    offset = f if offset is None else offset
    if offset < f or offset + f > base.freqs[0].shape[0] or h > base.freqs[1].shape[0] or w > base.freqs[2].shape[0]:
        raise ValueError("RoPE offset must separate streams and fit the precomputed frequency table")
    streams = []
    for start in (0, offset):
        streams.append(torch.cat((
            base.freqs[0][start:start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            base.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            base.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ), -1).reshape(f * h * w, 1, -1))
    return torch.cat(streams).to(device)


class Manifold4DModel(nn.Module):
    def __init__(self, base_model, positional_embedding_offset=31, cond_stream_t="honest"):
        super().__init__()
        if base_model.has_image_input or tuple(base_model.patch_size) != (1, 2, 2):
            raise ValueError("Manifold4D requires Wan2.1 T2V with patch size (1, 2, 2)")
        if cond_stream_t not in ("honest", "zero", "shared"):
            raise ValueError("cond_stream_t must be honest, zero or shared")
        if positional_embedding_offset is not None and positional_embedding_offset < 1:
            raise ValueError("RoPE offset must be positive or None")
        self.base_model = base_model
        self.positional_embedding_offset = positional_embedding_offset
        self.cond_stream_t = cond_stream_t
        self.vram_management_enabled = getattr(base_model, "vram_management_enabled", False)
        patch = base_model.patch_embedding
        storage_dtype = patch.weight.dtype
        if storage_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Manifold4D baseline supports fp32/fp16/bf16 weights; FP8 is not supported")
        # Make fresh native convs, even when the base conv is an offload wrapper.
        for name, channels, zero in (
            ("output_rgb_patch_embed", patch.weight.shape[1], False),
            ("output_anchor_patch_embed", 2, True),
            ("source_rgb_patch_embed", patch.weight.shape[1], False),
            ("source_mask_patch_embed", 2, True),
        ):
            conv = nn.Conv3d(channels, base_model.dim, (1, 2, 2), (1, 2, 2), dtype=storage_dtype)
            with torch.no_grad():
                if zero:
                    conv.weight.zero_()
                    conv.bias.zero_()
                else:
                    conv.weight.copy_(patch.weight.detach().cpu())
                    conv.bias.copy_(patch.bias.detach().cpu())
            setattr(self, name, conv)
        for block in base_model.blocks:
            target = unwrap(block)
            if hasattr(target, "cam_encoder") or hasattr(target, "projector"):
                raise ValueError("Use a clean Wan backbone, without Vista4D/JEPA modules")
            cam = nn.Linear(6, base_model.dim, dtype=storage_dtype)
            projector = nn.Linear(base_model.dim, base_model.dim, dtype=storage_dtype)
            nn.init.zeros_(cam.weight)
            nn.init.zeros_(cam.bias)
            nn.init.eye_(projector.weight)
            nn.init.zeros_(projector.bias)
            q = target.self_attn.q
            if isinstance(q, AutoWrappedLinear):
                config = {name: getattr(q, name) for name in (
                    "offload_dtype", "offload_device", "onload_dtype", "onload_device",
                    "preparing_dtype", "preparing_device", "computation_dtype", "computation_device", "vram_limit",
                )}
                cam = AutoWrappedLinear(cam, **config)
                projector = AutoWrappedLinear(projector, **config)
            target.cam_encoder = cam
            target.projector = projector
            target.self_attn.attn = ManifoldAttention(target.num_heads)
            target.cross_attn.attn = ManifoldAttention(target.num_heads)
            for name in ("norm1", "norm2", "norm3"):
                unwrap(getattr(target, name)).__class__ = ManifoldLayerNorm
        unwrap(base_model.head.norm).__class__ = ManifoldLayerNorm
        self.configure_trainable()

    @property
    def compute_dtype(self):
        return self.output_rgb_patch_embed.weight.dtype

    def configure_trainable(self):
        """Freeze backbone except Q/K/V/O, their RMSNorm, patchifies and camera/projector."""
        self.requires_grad_(False)
        for name in self.stream_names:
            getattr(self, name).requires_grad_(True)
        for block in self.base_model.blocks:
            block = unwrap(block)
            for module in (block.self_attn, block.cam_encoder, block.projector):
                module.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

    stream_names = (
        "output_rgb_patch_embed", "output_anchor_patch_embed",
        "source_rgb_patch_embed", "source_mask_patch_embed",
    )

    def _time_embeddings(self, timestep, source_timestep, tokens):
        base = self.base_model
        if self.cond_stream_t == "shared":
            source_timestep = timestep
        elif self.cond_stream_t == "zero" or source_timestep is None:
            source_timestep = torch.zeros_like(timestep)
        times = torch.stack((timestep, source_timestep), 1).float()
        # These frozen, small MLPs stay in fp32, including with CPU-offloaded
        # bf16 base weights. Avoid quantizing time to bf16 before the sinusoid.
        def linear(layer, value):
            return F.linear(value, layer.weight.to(value), layer.bias.to(value))
        with torch.autocast(device_type=times.device.type, enabled=False):
            emb = sinusoidal_embedding_1d(base.freq_dim, times.flatten()).float()
            emb = linear(base.time_embedding[2], F.silu(linear(base.time_embedding[0], emb)))
            modulation = linear(base.time_projection[1], F.silu(emb)).reshape(times.shape[0], 2, 6, base.dim)
            emb = emb.reshape(times.shape[0], 2, base.dim)
        emb_out = emb[:, :1].expand(-1, tokens, -1)
        modulation = torch.cat((modulation[:, :1].expand(-1, tokens, -1, -1),
                                modulation[:, 1:].expand(-1, tokens, -1, -1)), 1)
        return emb_out, modulation

    def forward(self, latents, timestep, context, *, source_latents, render_mask,
                source_mask, camera_embedding, source_timestep=None,
                use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        if latents.ndim != 5 or source_latents.shape != latents.shape:
            raise ValueError("Source/output latents must have the same [B, C, T, H, W] shape")
        b, _, f, h, w = latents.shape
        if h % 2 or w % 2 or render_mask.shape != (b, 2, f, h, w) or source_mask.shape != render_mask.shape:
            raise ValueError("Require even latent height/width and 2-channel coverage/motion masks")
        n = f * (h // 2) * (w // 2)
        if camera_embedding.shape != (b, 2 * n, 6):
            raise ValueError("Camera tokens must contain target rays followed by source rays")
        if timestep.shape != (b,) or context.shape[0] != b:
            raise ValueError("Latents, time and text must share the batch size")
        if source_timestep is not None and source_timestep.shape != (b,):
            raise ValueError("source_timestep must have shape [B]")
        base = self.base_model
        grid = (f, h // 2, w // 2)
        freqs = two_stream_freqs(base, grid, self.positional_embedding_offset, latents.device)
        with torch.autocast(device_type=latents.device.type, dtype=self.compute_dtype,
                            enabled=self.compute_dtype in (torch.float16, torch.bfloat16)):
            out = self.output_rgb_patch_embed(latents.to(self.compute_dtype)) + self.output_anchor_patch_embed(render_mask.to(self.compute_dtype))
            src = self.source_rgb_patch_embed(source_latents.to(self.compute_dtype)) + self.source_mask_patch_embed(source_mask.to(self.compute_dtype))
            x = torch.cat((out.flatten(2).transpose(1, 2), src.flatten(2).transpose(1, 2)), 1)
            time_emb, time_mod = self._time_embeddings(timestep, source_timestep, n)
            context = base.text_embedding(context.to(self.compute_dtype))
            for block in base.blocks:
                if use_gradient_checkpointing or use_gradient_checkpointing_offload:
                    manager = torch.autograd.graph.save_on_cpu() if use_gradient_checkpointing_offload else nullcontext()
                    with manager:
                        x = checkpoint(block, x, context, time_mod, freqs, camera_embedding, use_reentrant=False)
                else:
                    x = block(x, context, time_mod, freqs, camera_embedding)
            x = base.head(x[:, :n], time_emb)
        return base.unpatchify(x.float(), grid)

    def _flat_state(self):
        state = self.state_dict()
        return {flat: state[wrapped] for flat, wrapped in wrapped_key_map(self).items()}

    def load_checkpoint(self, directory):
        """Strictly load ALL three official artifacts; reject incomplete/Vista4D weights."""
        directory = Path(directory)
        filenames = ("conditioning_modules.pt", "camera_encoder.pt", "self_attn_full.pt")
        for name in filenames:
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Manifold4D checkpoint requires {directory / name}")
        mapping = wrapped_key_map(self)
        shapes = {k: tuple(v.shape) for k, v in self._flat_state().items()}

        def load_subset(state, expected, label):
            if set(state) != set(expected):
                raise ValueError(f"{label}: missing {sorted(set(expected) - set(state))}; unexpected {sorted(set(state) - set(expected))}")
            for name, value in state.items():
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != shapes[name]:
                    raise ValueError(f"{label}: shape mismatch for {name}")
            self.load_state_dict({mapping[k]: v for k, v in state.items()}, strict=False)

        state = torch.load(directory / filenames[0], map_location="cpu", weights_only=True)
        if state.get("schema_version", 1) != 1:
            raise ValueError("Unsupported conditioning_modules schema_version")
        modules = {}
        for name in self.stream_names:
            if name not in state:
                raise ValueError(f"conditioning_modules.pt is missing {name}")
            modules.update({f"{name}.{k}": v for k, v in state[name].items()})
        if "projectors" not in state:
            raise ValueError("conditioning_modules.pt is missing projectors")
        for key, value in state["projectors"].items():
            index, rest = key.split(".proj.")
            modules[f"base_model.blocks.{index}.projector.{rest}"] = value
        expected = [k for k in shapes if k.startswith(self.stream_names) or ".projector." in k]
        load_subset(modules, expected, filenames[0])
        del state, modules

        state = torch.load(directory / filenames[1], map_location="cpu", weights_only=True)
        if state.get("plucker_share_encoder", False) or "cam_encoders" not in state:
            raise ValueError("Released baseline requires per-block camera encoders")
        cameras = {}
        for key, value in state["cam_encoders"].items():
            index, rest = key.split(".proj.")
            cameras[f"base_model.blocks.{index}.cam_encoder.{rest}"] = value
        load_subset(cameras, [k for k in shapes if ".cam_encoder." in k], filenames[1])
        del state, cameras

        state = torch.load(directory / filenames[2], map_location="cpu", weights_only=True)
        attention = {f"base_model.{k}": v for k, v in state.items()}
        load_subset(attention, [k for k in shapes if ".self_attn." in k], filenames[2])
        return self

    def save_checkpoint(self, directory):
        """Export only the trainable subset in the official three-file schema."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        flat = self._flat_state()
        def subset(prefix):
            return {k[len(prefix):]: v.detach().cpu().contiguous() for k, v in flat.items() if k.startswith(prefix)}
        conditioning = {"schema_version": 1, **{name: subset(name + ".") for name in self.stream_names}}
        conditioning["projectors"] = {f"{i}.proj.{k}": v for i in range(len(self.base_model.blocks))
                                      for k, v in subset(f"base_model.blocks.{i}.projector.").items()}
        torch.save(conditioning, directory / "conditioning_modules.pt")
        cameras = {f"{i}.proj.{k}": v for i in range(len(self.base_model.blocks))
                   for k, v in subset(f"base_model.blocks.{i}.cam_encoder.").items()}
        torch.save({"plucker_share_encoder": False, "cam_encoders": cameras}, directory / "camera_encoder.pt")
        attention = {k[len("base_model."):]: v.detach().cpu().contiguous()
                     for k, v in flat.items() if ".self_attn." in k}
        torch.save(attention, directory / "self_attn_full.pt")
