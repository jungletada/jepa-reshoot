"""Small, zero-initialized residual cross-attention branches for Vista4D."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def spatial_positions(height, width, ref):
    y = (torch.arange(height, device=ref.device, dtype=ref.dtype) + .5) / height * 2 - 1
    x = (torch.arange(width, device=ref.device, dtype=ref.dtype) + .5) / width * 2 - 1
    return torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1).reshape(-1, 2)


@dataclass(frozen=True)
class AdapterConfig:
    feature_dim: int
    hidden_dim: int = 256
    num_heads: int = 8
    layers: tuple = (9, 19, 29, 39)
    spatial_pool: int = 8

    def validate(self, num_blocks):
        if min(self.feature_dim, self.hidden_dim, self.num_heads, self.spatial_pool) < 1:
            raise ValueError("Adapter dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("Adapter hidden_dim must be divisible by num_heads")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Adapter layers must be nonempty and unique")
        if min(self.layers) < 0 or max(self.layers) >= num_blocks:
            raise ValueError("Adapter layer index is outside this DiT")


class StructuralAttention(nn.Module):
    def __init__(self, dit_dim, config):
        super().__init__()
        self.norm = nn.LayerNorm(dit_dim)
        self.query = nn.Linear(dit_dim, config.hidden_dim)
        self.condition = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, config.hidden_dim))
        self.position = nn.Linear(2, config.hidden_dim)
        self.attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)
        self.output = nn.Linear(config.hidden_dim, dit_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, features, shape):
        b, _, d = x.shape
        t, h, w = shape
        _, c, _, kh, kw = features.shape
        query = self.query(self.norm(x.reshape(b * t, h * w, d).to(self.norm.weight.dtype)))
        query = query + self.position(spatial_positions(h, w, query))
        tokens = features.permute(0, 2, 3, 4, 1).reshape(b * t, kh * kw, c)
        values = self.condition(tokens.to(self.norm.weight.dtype))
        keys = values + self.position(spatial_positions(kh, kw, values))
        result, _ = self.attention(query, keys, values, need_weights=False)
        return self.output(result).reshape_as(x).to(x)


class JEPAAdapter(nn.Module):
    def __init__(self, dit_dim, num_blocks, config):
        super().__init__()
        config.validate(num_blocks)
        self.config, self.dit_dim = config, dit_dim
        self.branches = nn.ModuleDict({str(i): StructuralAttention(dit_dim, config) for i in config.layers})

    def prepare(self, features, shape, batch_size):
        t, _, _ = shape
        if features.ndim != 5 or features.shape[1] != self.config.feature_dim or features.shape[2] != t:
            raise ValueError("JEPA features must be B,C,T,H,W, aligned to the VAE latent frame coordinates")
        if features.shape[0] == 1:
            features = features.expand(batch_size, -1, -1, -1, -1)
        if features.shape[0] != batch_size:
            raise ValueError("JEPA feature batch does not match DiT tokens")
        # Bound attention memory. Keep time separate and preserve the full field of view.
        h, w = (min(v, self.config.spatial_pool) for v in features.shape[-2:])
        return F.adaptive_avg_pool3d(features, (t, h, w))

    def forward(self, x, features, shape, layer, scale=1.0):
        n = shape[0] * shape[1] * shape[2]
        output = x[:, :n] + scale * self.branches[str(layer)](x[:, :n], features, shape)
        return torch.cat((output, x[:, n:]), dim=1)

    def save(self, path, encoder_id):
        torch.save({"format": "jepa-adapter-v1", "config": asdict(self.config),
                    "dit_dim": self.dit_dim, "encoder_id": encoder_id,
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}, path)

    @classmethod
    def load(cls, path, dit_dim, num_blocks):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != "jepa-adapter-v1" or data["dit_dim"] != dit_dim:
            raise ValueError("Adapter checkpoint is incompatible with this DiT")
        adapter = cls(dit_dim, num_blocks, AdapterConfig(**data["config"]))
        adapter.load_state_dict(data["state_dict"], strict=True)
        return adapter, data["encoder_id"]
