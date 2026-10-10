"""Dense feature caches with explicit provenance and temporal coordinates."""
from dataclasses import dataclass
from pathlib import Path

import torch


ROLES = {"source", "projected", "oracle", "predicted"}


def resample_time(features, positions, query):
    """Linearly sample B,C,T,H,W at frame coordinates; clamp at clip edges."""
    positions = positions.to(device=features.device, dtype=torch.float32)
    query = query.to(device=features.device, dtype=torch.float32)
    if positions.ndim != 1 or positions.numel() != features.shape[2]:
        raise ValueError("Temporal positions must match the feature grid")
    if not torch.isfinite(positions).all() or (positions.diff() <= 0).any():
        raise ValueError("Temporal positions must be finite and strictly increasing")
    if positions.numel() == 1:
        return features.expand(-1, -1, query.numel(), -1, -1)
    query = query.clamp(positions[0], positions[-1])
    right = torch.searchsorted(positions, query).clamp(1, positions.numel() - 1)
    left = right - 1
    weight = ((query - positions[left]) / (positions[right] - positions[left])).to(features.dtype)
    return features[:, :, left].lerp(features[:, :, right], weight[None, None, :, None, None])


@dataclass
class FeatureCache:
    features: torch.Tensor  # B,C,T,H,W, dense final-layer teacher features
    frame_positions: torch.Tensor  # T, centers in original clip frame coordinates
    num_frames: int
    image_size: tuple  # original prepared video (height, width), before teacher resize
    encoder_id: str  # architecture + checkpoint digest + preprocessing version
    role: str

    def validate(self):
        if self.role not in ROLES or not self.encoder_id:
            raise ValueError("A known feature role and nonempty encoder_id are required")
        if self.features.ndim != 5 or min(self.features.shape) < 1:
            raise ValueError("Features must have shape B,C,T,H,W")
        if not self.features.is_floating_point() or not torch.isfinite(self.features).all():
            raise ValueError("Features must be finite floating-point values")
        if self.num_frames < 1 or len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError("Invalid clip dimensions")
        resample_time(self.features, self.frame_positions, self.frame_positions)
        if (self.frame_positions < 0).any() or (self.frame_positions > self.num_frames - 1).any():
            raise ValueError("Feature positions fall outside the clip")
        return self

    def save(self, path):
        self.validate()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "jepa-features-v1", **{
            **vars(self), "features": self.features.detach().cpu(),
            "frame_positions": self.frame_positions.detach().cpu(),
        }}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.pop("format", None) != "jepa-features-v1":
            raise ValueError("Unsupported JEPA feature cache")
        return cls(**data).validate()

    def for_generation(self, *, mode, num_frames, image_size, batch_size, encoder_id):
        if mode != self.role:
            raise ValueError(f"Requested {mode} features, but cache role is {self.role}")
        if self.encoder_id != encoder_id:
            raise ValueError("Feature encoder differs from the trained adapter's encoder")
        if self.num_frames != num_frames or tuple(self.image_size) != tuple(image_size):
            raise ValueError("JEPA cache must match the prepared video length and resolution")
        if (num_frames - 1) % 4:
            raise ValueError("Vista4D requires a 4n+1 frame clip")
        features = resample_time(self.features, self.frame_positions, torch.arange(0, num_frames, 4))
        if features.shape[0] == 1:
            features = features.expand(batch_size, -1, -1, -1, -1)
        if features.shape[0] != batch_size:
            raise ValueError("JEPA cache batch does not match the video batch")
        return features
