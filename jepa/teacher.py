"""Frozen local V-JEPA teacher; no network downloads or temporal subsampling."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .features import FeatureCache


class FrozenVideoTeacher(nn.Module):
    def __init__(self, encoder, encoder_id, size=(224, 384), patch_size=16, tubelet_size=2):
        super().__init__()
        if min(*size, patch_size, tubelet_size) < 1 or any(v % patch_size for v in size):
            raise ValueError("Teacher image size must be positive and divisible by patch_size")
        self.encoder = encoder.eval().requires_grad_(False)
        self.encoder_id = f"{encoder_id}:rgb-imagenet-resize-v1:{size[0]}x{size[1]}:p{patch_size}:t{tubelet_size}"
        self.size, self.patch_size, self.tubelet_size = size, patch_size, tubelet_size

    @torch.no_grad()
    def encode(self, video, role):
        if video.ndim != 5 or video.shape[1] != 3 or not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError("Teacher input must be B,3,T,H,W RGB in [0,1]")
        b, _, frames, ih, iw = video.shape
        images = video.permute(0, 2, 1, 3, 4).reshape(b * frames, 3, ih, iw)
        images = F.interpolate(images.float(), self.size, mode="bilinear", align_corners=False, antialias=True)
        video = images.reshape(b, frames, 3, *self.size).permute(0, 2, 1, 3, 4)
        padding = (-frames) % self.tubelet_size
        if padding:
            video = torch.cat((video, video[:, :, -1:].expand(-1, -1, padding, -1, -1)), dim=2)
        mean = video.new_tensor([.485, .456, .406])[None, :, None, None, None]
        std = video.new_tensor([.229, .224, .225])[None, :, None, None, None]
        self.encoder.eval()
        tokens = self.encoder((video - mean) / std)
        t, h, w = video.shape[2] // self.tubelet_size, self.size[0] // self.patch_size, self.size[1] // self.patch_size
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[:2] != (b, t * h * w):
            raise ValueError("Teacher must return dense final-layer B,(T*H*W),C tokens without CLS/register tokens")
        features = tokens.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
        positions = (torch.arange(t).float() * self.tubelet_size + (self.tubelet_size - 1) / 2).clamp(max=frames - 1)
        return FeatureCache(features.cpu(), positions, frames, (ih, iw), self.encoder_id, role).validate()


def load_local_teacher(repo, checkpoint, architecture, checkpoint_key, size):
    # Official hub factories return (encoder, pretraining predictor); the latter
    # predicts masked tokens and is not the novel-view predictor in this project.
    encoder, _ = torch.hub.load(str(repo), architecture, source="local", pretrained=False)
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = data[checkpoint_key]
    cleaned = {}
    for name, value in state.items():
        for prefix in ("module.", "backbone."):
            name = name.removeprefix(prefix)
        cleaned[name] = value
    encoder.load_state_dict(cleaned, strict=True)
    digest = hashlib.sha256()
    with open(checkpoint, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return FrozenVideoTeacher(encoder, f"{architecture}:{digest.hexdigest()}:{checkpoint_key}", size,
                              patch_size=encoder.patch_size, tubelet_size=encoder.tubelet_size)
