import argparse
from typing import Any, Dict, List, Optional

import torch


# Mirrors WanGP's "run with lower VRAM" mode, built on diffsynth's existing
# offload (AutoWrappedModule block-swap) and FP8 compute support instead of a
# separately quantized checkpoint.
PRESETS: Dict[str, Dict[str, Any]] = {
    "full": {
        "offload_device": None,
        "offload_dtype": None,
        "fp8_compute": False,
        "vram_limit_gb": None,
    },
    "balanced": {
        "offload_device": "cpu",
        "offload_dtype": torch.bfloat16,
        "fp8_compute": False,
        "vram_limit_gb": 16.0,
    },
    "low_vram": {
        "offload_device": "cpu",
        "offload_dtype": torch.bfloat16,
        "fp8_compute": True,
        "vram_limit_gb": 10.0,
    },
}

# Only the DiT (the 14B transformer) is run in FP8; the text encoder and VAE
# are small enough that quantizing them isn't worth the quality risk.
DIT_FILE_PATTERN = "diffusion_pytorch_model"


def add_vram_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--vram_preset", choices=sorted(PRESETS), default="full",
        help=(
            "WanGP-style VRAM preset. 'full' keeps the previous behavior (everything resident "
            "on GPU in bf16). 'balanced' offloads idle model blocks to CPU between forward passes. "
            "'low_vram' additionally runs the DiT's compute in FP8."
        ),
    )
    parser.add_argument(
        "--vram_limit", type=float, default=None,
        help=(
            "GPU memory budget in GB used to decide how many blocks stay resident. "
            "Overrides the selected --vram_preset's default; ignored when --vram_preset=full."
        ),
    )
    parser.add_argument(
        "--fp8_compute", action=argparse.BooleanOptionalAction, default=None,
        help="Run the DiT's compute in FP8. Overrides the selected --vram_preset's default.",
    )
    return parser


def resolve_vram_settings(args: argparse.Namespace) -> Dict[str, Any]:
    preset = PRESETS[args.vram_preset]
    vram_limit = args.vram_limit if args.vram_limit is not None else preset["vram_limit_gb"]
    fp8_compute = args.fp8_compute if args.fp8_compute is not None else preset["fp8_compute"]
    return {
        "offload_device": preset["offload_device"],
        "offload_dtype": preset["offload_dtype"],
        "fp8_compute": fp8_compute,
        "vram_limit": vram_limit,
    }


def apply_vram_settings_to_model_configs(model_configs: List, model_ids: List[str], settings: Dict[str, Any]) -> List:
    """Applies `settings` (from `resolve_vram_settings`) to each ModelConfig in place.

    `model_ids` must be the same `id:origin_file_pattern` strings used to build `model_configs`,
    so FP8 compute can be scoped to the DiT weights only.
    """
    for model_config, model_id in zip(model_configs, model_ids):
        model_config.offload_device = settings["offload_device"]
        model_config.offload_dtype = settings["offload_dtype"]
        if settings["fp8_compute"] and DIT_FILE_PATTERN in model_id:
            model_config.computation_dtype = torch.float8_e4m3fn
    return model_configs
