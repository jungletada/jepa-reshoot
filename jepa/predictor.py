"""Source-only world memory and target-camera dense feature predictor."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _rays(c2w, intrinsics, grid, image_size):
    """OpenCV camera convention: x right, y down, z forward; intrinsics in pixels."""
    h, w = grid
    ih, iw = image_size
    if c2w.shape[-2:] != (4, 4) or intrinsics.shape != (*c2w.shape[:2], 4):
        raise ValueError("Expected cameras B,T,4,4 and intrinsics B,T,4 (fx,fy,cx,cy)")
    if not torch.isfinite(c2w).all() or not torch.isfinite(intrinsics).all() or (intrinsics[..., :2] <= 0).any():
        raise ValueError("Cameras must be finite with positive focal lengths")
    yy, xx = torch.meshgrid(
        (torch.arange(h, device=c2w.device) + .5) * ih / h,
        (torch.arange(w, device=c2w.device) + .5) * iw / w, indexing="ij")
    fx, fy, cx, cy = (v[..., None, None] for v in intrinsics.unbind(-1))
    local = torch.stack(((xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx - cx)), dim=-1)
    directions = torch.einsum("btij,bthwj->bthwi", c2w[..., :3, :3], local)
    origins = c2w[..., :3, 3][:, :, None, None].expand_as(directions)
    return origins, directions


def prepare_geometry(source_cache, geometry):
    """Use source RGB-D/cameras plus target cameras and SOURCE-rendered visibility.

    The target RGB/depth/teacher features are deliberately not arguments.
    Geometry cameras/depth must share one world frame and the prepared clip's FOV.
    """
    b, _, t, h, w = source_cache.features.shape
    frames = source_cache.num_frames
    required = {"source_depth", "source_c2w", "source_intrinsics", "target_c2w", "target_intrinsics", "target_visibility"}
    if set(geometry) != required:
        raise ValueError(f"Geometry must contain exactly {sorted(required)} (no target ground truth)")
    g = {k: v.float() for k, v in geometry.items()}
    if any(v.shape[:2] != (b, frames) for v in g.values()):
        raise ValueError("Geometry batch/frame counts must match the source feature cache")
    ids = source_cache.frame_positions.round().long()
    depth = g["source_depth"][:, ids]
    if depth.ndim != 4 or tuple(depth.shape[-2:]) != tuple(source_cache.image_size):
        raise ValueError("Source depth must be B,T,H,W at the prepared video resolution")
    depth = F.interpolate(depth.reshape(b * t, 1, *depth.shape[-2:]), (h, w), mode="nearest-exact").reshape(b, t, h, w)
    valid = torch.isfinite(depth) & (depth > 0)
    if not valid.flatten(1).any(1).all():
        raise ValueError("Each example needs at least one valid source depth")
    depth = torch.where(valid, depth, 0)
    src_origin, src_direction = _rays(g["source_c2w"][:, ids], g["source_intrinsics"][:, ids], (h, w), source_cache.image_size)
    xyz = src_origin + src_direction * depth[..., None]
    tgt_origin, tgt_direction = _rays(g["target_c2w"][:, ids], g["target_intrinsics"][:, ids], (h, w), source_cache.image_size)
    # Canonical source-first camera frame, scaled using SOURCE depths only.
    anchor = g["source_c2w"][:, 0]
    rotation = anchor[:, :3, :3].transpose(-1, -2)
    translation = anchor[:, :3, 3][:, None, None, None]
    scale = torch.stack([depth[i][valid[i]].median() for i in range(b)]).clamp_min(1e-6)
    def canonical(points):
        return torch.einsum("bij,bthwj->bthwi", rotation, points - translation) / scale[:, None, None, None, None]
    xyz = canonical(xyz).masked_fill(~valid[..., None], 0)
    origins = canonical(tgt_origin)
    directions = F.normalize(torch.einsum("bij,bthwj->bthwi", rotation, tgt_direction), dim=-1)
    visibility = g["target_visibility"][:, ids]
    if (visibility.ndim != 4 or tuple(visibility.shape[-2:]) != tuple(source_cache.image_size)
            or not torch.isfinite(visibility).all() or (visibility < 0).any() or (visibility > 1).any()):
        raise ValueError("Target visibility must be a finite source-rendered mask in [0,1]")
    visibility = F.interpolate(visibility.reshape(b * t, 1, *visibility.shape[-2:]), (h, w), mode="area").reshape(b, t, h, w)
    times = source_cache.frame_positions.float() / max(frames - 1, 1)
    return {"source_xyz": xyz.permute(0, 4, 1, 2, 3), "source_valid": valid,
            "target_rays": torch.cat((origins, directions), dim=-1).permute(0, 4, 1, 2, 3),
            "query_visibility": visibility, "times": times}


@dataclass(frozen=True)
class PredictorConfig:
    feature_dim: int
    hidden_dim: int = 256
    num_heads: int = 8
    memory_tokens: int = 128
    depth: int = 2


class TargetFeaturePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        if min(asdict(config).values()) < 1 or config.hidden_dim % config.num_heads:
            raise ValueError("Positive predictor dimensions and divisible attention heads are required")
        self.config = config
        d = config.hidden_dim
        self.source_norm = nn.LayerNorm(config.feature_dim)
        self.source = nn.Linear(config.feature_dim + 4, d)  # xyz and time
        self.query = nn.Linear(8, d)  # ray origin/direction, time, visibility
        self.slots = nn.Parameter(torch.randn(1, config.memory_tokens, d) * .02)
        self.aggregate = nn.MultiheadAttention(d, config.num_heads, batch_first=True)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(d, config.num_heads, d * 4, dropout=0, batch_first=True, norm_first=True)
            for _ in range(config.depth)
        ])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.feature_dim))

    def forward(self, source_features, source_xyz, source_valid, target_rays, query_visibility, times):
        b, c, t, h, w = source_features.shape
        if source_xyz.shape != (b, 3, t, h, w) or target_rays.shape != (b, 6, t, h, w):
            raise ValueError("Source xyz and target rays must share the source feature grid")
        if source_valid.shape != (b, t, h, w) or query_visibility.shape != source_valid.shape or times.shape != (t,):
            raise ValueError("Invalid predictor mask or time shape")
        if not source_valid.flatten(1).any(1).all():
            raise ValueError("Each example needs a valid source memory token")
        flatten = lambda x: x.permute(0, 2, 3, 4, 1).reshape(b, -1, x.shape[1])
        time = times[None, :, None, None, None].expand(b, t, h, w, 1).reshape(b, -1, 1)
        source = self.source(torch.cat((self.source_norm(flatten(source_features)), flatten(source_xyz), time), dim=-1))
        memory, _ = self.aggregate(self.slots.expand(b, -1, -1), source, source,
                                   key_padding_mask=~source_valid.bool().flatten(1), need_weights=False)
        query = self.query(torch.cat((flatten(target_rays), time, query_visibility.reshape(b, -1, 1)), dim=-1))
        # Query self-attention would be quadratic in dense video tokens. Decode each
        # time slice separately; all slices still attend to the full world memory.
        query = query.reshape(b * t, h * w, -1)
        memory = memory[:, None].expand(-1, t, -1, -1).reshape(b * t, self.config.memory_tokens, -1)
        for layer in self.decoder:
            query = layer(query, memory)
        return self.output(query).reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)

    def save(self, path, encoder_id):
        torch.save({"format": "jepa-predictor-v1", "config": asdict(self.config), "encoder_id": encoder_id,
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != "jepa-predictor-v1":
            raise ValueError("Unsupported predictor checkpoint")
        model = cls(PredictorConfig(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return model, data["encoder_id"]


def feature_regression_loss(prediction, teacher, weights=None):
    """Channel-normalized L2, teacher stop-gradient, optionally weight occlusions."""
    if prediction.shape != teacher.shape:
        raise ValueError("Prediction and teacher grids must match")
    normalize = lambda x: F.layer_norm(x.movedim(1, -1).float(), (x.shape[1],))
    error = (normalize(prediction) - normalize(teacher.detach())).square().mean(-1)
    if weights is None:
        return error.mean()
    if weights.shape != error.shape or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Regression weights must match B,T,H,W and have positive total weight")
    return (error * weights).sum() / weights.sum()
