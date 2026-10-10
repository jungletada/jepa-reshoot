from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch

from utils.split_manifest import validate_flowlong_window_manifest


@dataclass(frozen=True)
class FlowLongGeometry:
    pixel_window: int
    pixel_stride: int
    pixel_overlap: int
    temporal_factor: int
    latent_window: int
    latent_stride: int
    latent_overlap: int
    pixel_starts: Tuple[int, ...]
    latent_starts: Tuple[int, ...]
    global_latent_frames: int
    padded_pixel_frames: int
    valid_pixel_frames: int
    trim_right: int

    @property
    def num_windows(self) -> int:
        return len(self.pixel_starts)


@dataclass(frozen=True)
class FlowLongSamplingConfig:
    stochastic_threshold: float = 0.6
    stochastic_enabled: bool = True
    microbatch_size: int = 1
    matching_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.stochastic_enabled and not 0.0 < self.stochastic_threshold <= 1.0:
            raise ValueError(
                "stochastic_threshold must satisfy 0 < threshold <= 1 when "
                f"stochastic sampling is enabled, got {self.stochastic_threshold}"
            )
        if self.microbatch_size <= 0:
            raise ValueError(
                f"microbatch_size must be positive, got {self.microbatch_size}"
            )
        if not self.matching_dtype.is_floating_point:
            raise ValueError(
                f"matching_dtype must be floating point, got {self.matching_dtype}"
            )


def build_geometry_from_manifest(
    manifest: Dict,
    temporal_factor: int = 4,
) -> FlowLongGeometry:
    clips = validate_flowlong_window_manifest(
        manifest,
        clip_frames=49,
        overlap=25,
        temporal_alignment=temporal_factor,
    )
    pixel_window = int(manifest["clip_frames"])
    pixel_overlap = int(manifest["overlap"])
    pixel_stride = int(manifest["stride"])
    if (pixel_window - 1) % temporal_factor:
        raise ValueError(
            f"pixel_window={pixel_window} is incompatible with temporal_factor="
            f"{temporal_factor}"
        )
    if pixel_stride % temporal_factor:
        raise ValueError(
            f"pixel_stride={pixel_stride} is incompatible with temporal_factor="
            f"{temporal_factor}"
        )

    latent_window = (pixel_window - 1) // temporal_factor + 1
    latent_stride = pixel_stride // temporal_factor
    latent_overlap = latent_window - latent_stride
    if latent_overlap < latent_stride:
        raise ValueError(
            "FlowLong requires latent overlap O >= latent stride S, got "
            f"F={latent_window}, S={latent_stride}, O={latent_overlap}"
        )

    pixel_starts = tuple(int(clip["start_frame"]) for clip in clips)
    first_pixel_start = pixel_starts[0]
    latent_starts = tuple(
        (start - first_pixel_start) // temporal_factor for start in pixel_starts
    )
    global_latent_frames = latent_starts[-1] + latent_window
    padded_pixel_frames = 1 + temporal_factor * (global_latent_frames - 1)
    manifest_end = int(manifest["end_exclusive"])
    valid_pixel_frames = manifest_end - first_pixel_start
    trim_right = padded_pixel_frames - valid_pixel_frames
    if trim_right < 0:
        raise ValueError(
            f"Global latent covers {padded_pixel_frames} pixels but manifest requires "
            f"{valid_pixel_frames}"
        )

    return FlowLongGeometry(
        pixel_window=pixel_window,
        pixel_stride=pixel_stride,
        pixel_overlap=pixel_overlap,
        temporal_factor=temporal_factor,
        latent_window=latent_window,
        latent_stride=latent_stride,
        latent_overlap=latent_overlap,
        pixel_starts=pixel_starts,
        latent_starts=latent_starts,
        global_latent_frames=global_latent_frames,
        padded_pixel_frames=padded_pixel_frames,
        valid_pixel_frames=valid_pixel_frames,
        trim_right=trim_right,
    )


def linear_blend_weights(
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if overlap < 2:
        raise ValueError(f"FlowLong linear blending requires overlap >= 2, got {overlap}")
    return torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)


def _validate_window_tensor(
    window_values: torch.Tensor,
    geometry: FlowLongGeometry,
) -> None:
    if window_values.ndim < 3:
        raise ValueError(
            "window_values must have shape [window, channel, frame, ...], got "
            f"{tuple(window_values.shape)}"
        )
    if window_values.shape[0] != geometry.num_windows:
        raise ValueError(
            f"Expected {geometry.num_windows} windows, got {window_values.shape[0]}"
        )
    if window_values.shape[2] != geometry.latent_window:
        raise ValueError(
            f"Expected latent window length {geometry.latent_window}, got "
            f"{window_values.shape[2]}"
        )


def aggregate_window_values(
    window_values: torch.Tensor,
    geometry: FlowLongGeometry,
    calculation_dtype: torch.dtype = torch.float32,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Aggregate regular windows with rightmost-pair/last-writer-wins blending."""
    _validate_window_tensor(window_values, geometry)
    values = window_values.to(dtype=calculation_dtype)
    output_shape = (
        1,
        values.shape[1],
        geometry.global_latent_frames,
        *values.shape[3:],
    )
    global_values = torch.empty(
        output_shape,
        dtype=calculation_dtype,
        device=values.device,
    )
    stride = geometry.latent_stride
    overlap = geometry.latent_overlap

    global_values[0, :, :stride] = values[0, :, :stride]
    weights = linear_blend_weights(
        overlap,
        device=values.device,
        dtype=calculation_dtype,
    )
    weights = weights.view(1, overlap, *([1] * (values.ndim - 3)))
    for pair_index in range(geometry.num_windows - 1):
        global_start = geometry.latent_starts[pair_index + 1]
        left = values[pair_index, :, stride : stride + overlap]
        right = values[pair_index + 1, :, :overlap]
        blend = (1.0 - weights) * left + weights * right
        global_values[0, :, global_start : global_start + overlap] = blend

    suffix_start = geometry.latent_starts[-1] + overlap
    global_values[0, :, suffix_start:] = values[-1, :, overlap:]
    return global_values.to(dtype=output_dtype or calculation_dtype)


def aggregate_predicted_clean(
    window_x0: torch.Tensor,
    geometry: FlowLongGeometry,
    matching_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return aggregate_window_values(
        window_x0,
        geometry,
        calculation_dtype=matching_dtype,
        output_dtype=matching_dtype,
    )


def slice_global_latents(
    global_latents: torch.Tensor,
    geometry: FlowLongGeometry,
) -> torch.Tensor:
    if global_latents.ndim < 3 or global_latents.shape[0] != 1:
        raise ValueError(
            "global_latents must have shape [1, channel, frame, ...], got "
            f"{tuple(global_latents.shape)}"
        )
    if global_latents.shape[2] != geometry.global_latent_frames:
        raise ValueError(
            f"Expected {geometry.global_latent_frames} global latent frames, got "
            f"{global_latents.shape[2]}"
        )
    return torch.cat(
        [
            global_latents[
                :,
                :,
                start : start + geometry.latent_window,
            ]
            for start in geometry.latent_starts
        ],
        dim=0,
    )


def overlap_error(
    window_values: torch.Tensor,
    geometry: FlowLongGeometry,
) -> Dict[str, float]:
    _validate_window_tensor(window_values, geometry)
    total = 0.0
    count = 0
    maximum = 0.0
    stride = geometry.latent_stride
    overlap = geometry.latent_overlap
    for pair_index in range(geometry.num_windows - 1):
        left = window_values[pair_index, :, stride : stride + overlap]
        right = window_values[pair_index + 1, :, :overlap]
        difference = (left.float() - right.float()).abs()
        total += float(difference.sum().item())
        count += difference.numel()
        maximum = max(maximum, float(difference.max().item()))
    return {
        "mae": total / count,
        "max_abs": maximum,
    }


def flowlong_next_state(
    global_xt: torch.Tensor,
    global_x0: torch.Tensor,
    t: float,
    s: float,
    stochastic: bool,
    generator: Optional[torch.Generator] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if global_xt.shape != global_x0.shape:
        raise ValueError(
            f"global_xt shape {tuple(global_xt.shape)} does not match global_x0 "
            f"shape {tuple(global_x0.shape)}"
        )
    if not 0.0 <= s <= t <= 1.0:
        raise ValueError(f"Expected 0 <= s <= t <= 1, got t={t}, s={s}")
    dtype = output_dtype or global_xt.dtype
    xt = global_xt.float()
    x0 = global_x0.float()
    if s == 0.0:
        return x0.to(dtype=dtype)
    if stochastic:
        if generator is None:
            raise ValueError("A generator is required for stochastic FlowLong updates")
        noise = torch.randn(
            x0.shape,
            generator=generator,
            device=generator.device,
            dtype=torch.float32,
        ).to(device=x0.device)
        next_state = (1.0 - s) * x0 + s * noise
    else:
        if t <= 0.0:
            raise ValueError("Deterministic FlowLong update requires t > 0")
        x1 = (xt - (1.0 - t) * x0) / t
        next_state = (1.0 - s) * x0 + s * x1
    return next_state.to(dtype=dtype)


def run_flowlong_denoising(
    initial_window_latents: torch.Tensor,
    geometry: FlowLongGeometry,
    sigmas: torch.Tensor,
    predict_velocity: Callable[[int, torch.Tensor], torch.Tensor],
    config: FlowLongSamplingConfig,
    stochastic_generator: Optional[torch.Generator],
    step_indices: Optional[Iterable[int]] = None,
    synchronize: Optional[Callable[[], None]] = None,
) -> Tuple[torch.Tensor, list[Dict]]:
    """Run the synchronized FlowLong state loop around a velocity callback."""
    _validate_window_tensor(initial_window_latents, geometry)
    if sigmas.ndim != 1 or len(sigmas) == 0:
        raise ValueError(f"sigmas must be a non-empty 1D tensor, got {sigmas}")
    state_dtype = initial_window_latents.dtype
    global_state = None
    reports = []
    indices = range(len(sigmas)) if step_indices is None else step_indices

    for step_index in indices:
        if synchronize is not None:
            synchronize()
        started = perf_counter()
        t = float(sigmas[step_index].item())
        s = 0.0 if step_index + 1 == len(sigmas) else float(sigmas[step_index + 1].item())
        if global_state is None:
            window_xt = initial_window_latents
            global_xt = aggregate_window_values(
                window_xt,
                geometry,
                calculation_dtype=config.matching_dtype,
                output_dtype=config.matching_dtype,
            )
        else:
            window_xt = slice_global_latents(global_state, geometry)
            global_xt = global_state.to(dtype=config.matching_dtype)

        velocity = predict_velocity(step_index, window_xt)
        if velocity.shape != window_xt.shape:
            raise ValueError(
                f"Velocity shape {tuple(velocity.shape)} does not match window state "
                f"shape {tuple(window_xt.shape)} at step {step_index}"
            )
        window_x0 = (
            window_xt.to(dtype=config.matching_dtype)
            - t * velocity.to(dtype=config.matching_dtype)
        )
        before = overlap_error(window_x0, geometry)
        global_x0 = aggregate_predicted_clean(
            window_x0,
            geometry,
            matching_dtype=config.matching_dtype,
        )
        after = overlap_error(slice_global_latents(global_x0, geometry), geometry)
        stochastic = config.stochastic_enabled and t >= config.stochastic_threshold
        global_state = flowlong_next_state(
            global_xt,
            global_x0,
            t=t,
            s=s,
            stochastic=stochastic,
            generator=stochastic_generator,
            output_dtype=state_dtype,
        )
        if synchronize is not None:
            synchronize()
        reports.append(
            {
                "step": int(step_index),
                "sigma_t": t,
                "sigma_s": s,
                "stochastic": stochastic and s > 0.0,
                "overlap_before_mae": before["mae"],
                "overlap_before_max_abs": before["max_abs"],
                "overlap_after_max_abs": after["max_abs"],
                "elapsed_seconds": perf_counter() - started,
            }
        )

    if global_state is None:
        raise ValueError("No FlowLong denoising steps were executed")
    return global_state, reports
