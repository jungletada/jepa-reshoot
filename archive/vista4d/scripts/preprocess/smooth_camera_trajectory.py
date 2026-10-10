from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def ensure_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    quats = quats.copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] *= -1.0
    return quats


def smooth_rotations(c2w: np.ndarray, sigma: float, mode: str) -> np.ndarray:
    rotations = Rotation.from_matrix(c2w[:, :3, :3])
    quats = ensure_quaternion_continuity(rotations.as_quat())

    # Smooth in quaternion space, then renormalize and project back to SO(3).
    smoothed_quats = gaussian_filter1d(quats, sigma=sigma, axis=0, mode=mode)
    norms = np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Encountered near-zero quaternion after smoothing.")
    smoothed_quats = smoothed_quats / norms
    return Rotation.from_quat(smoothed_quats).as_matrix()


def reanchor_to_first_pose(smoothed: np.ndarray, reference_first: np.ndarray) -> np.ndarray:
    """Keep the smoothed relative motion but start from the original first camera."""
    anchored = smoothed.copy()
    correction = reference_first @ np.linalg.inv(smoothed[0])
    anchored = correction[None, :, :] @ anchored
    anchored[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return anchored


def relative_rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    value = (np.trace(rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def summarize(c2w: np.ndarray) -> dict:
    translations = c2w[:, :3, 3]
    delta = translations - translations[0]
    step_t = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    rel_rot = np.array([relative_rotation_angle_deg(c2w[0], c2w[i]) for i in range(len(c2w))])
    step_rot = np.array([relative_rotation_angle_deg(c2w[i - 1], c2w[i]) for i in range(1, len(c2w))])
    return {
        "frames": int(len(c2w)),
        "translation_first": translations[0].tolist(),
        "translation_last": translations[-1].tolist(),
        "translation_delta_last": delta[-1].tolist(),
        "translation_axis_min": translations.min(axis=0).tolist(),
        "translation_axis_max": translations.max(axis=0).tolist(),
        "translation_axis_span": (translations.max(axis=0) - translations.min(axis=0)).tolist(),
        "translation_max_norm_from_first": float(np.linalg.norm(delta, axis=1).max()),
        "translation_path_length": float(step_t.sum()),
        "translation_step_mean": float(step_t.mean()) if len(step_t) else 0.0,
        "translation_step_max": float(step_t.max()) if len(step_t) else 0.0,
        "rotation_rel_max_deg": float(rel_rot.max()) if len(rel_rot) else 0.0,
        "rotation_rel_last_deg": float(rel_rot[-1]) if len(rel_rot) else 0.0,
        "rotation_step_mean_deg": float(step_rot.mean()) if len(step_rot) else 0.0,
        "rotation_step_max_deg": float(step_rot.max()) if len(step_rot) else 0.0,
    }


def print_summary(name: str, stats: dict) -> None:
    print(f"[{name}]")
    print(f"  frames: {stats['frames']}")
    print(f"  translation first: {np.array(stats['translation_first'])}")
    print(f"  translation last:  {np.array(stats['translation_last'])}")
    print(f"  translation delta: {np.array(stats['translation_delta_last'])}")
    print(f"  translation span:  {np.array(stats['translation_axis_span'])}")
    print(f"  max norm from first: {stats['translation_max_norm_from_first']:.6f}")
    print(f"  path length:         {stats['translation_path_length']:.6f}")
    print(f"  step mean/max:       {stats['translation_step_mean']:.6f} / {stats['translation_step_max']:.6f}")
    print(f"  rot rel max/last:    {stats['rotation_rel_max_deg']:.6f} / {stats['rotation_rel_last_deg']:.6f} deg")
    print(f"  rot step mean/max:   {stats['rotation_step_mean_deg']:.6f} / {stats['rotation_step_max_deg']:.6f} deg")


def main(args):
    repo_root = Path(__file__).resolve().parents[2]
    input_path = resolve_path(args.input, repo_root)
    output_path = resolve_path(args.output, repo_root) if args.output else input_path.with_name("cameras_gaussian_smooth.npz")

    data = np.load(input_path)
    c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected cam_c2w shape [N,4,4], got {c2w.shape}")
    if intrinsics.ndim != 2 or intrinsics.shape[0] != c2w.shape[0]:
        raise ValueError(f"Expected intrinsics shape [N,4], got {intrinsics.shape}")

    smoothed = c2w.copy()
    if args.translation_sigma > 0:
        smoothed[:, :3, 3] = gaussian_filter1d(
            c2w[:, :3, 3],
            sigma=args.translation_sigma,
            axis=0,
            mode=args.mode,
        )
    if args.rotation_sigma > 0:
        smoothed[:, :3, :3] = smooth_rotations(c2w, sigma=args.rotation_sigma, mode=args.mode)
    smoothed[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    if args.anchor_first:
        smoothed = reanchor_to_first_pose(smoothed, c2w[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "cam_c2w": smoothed.astype(np.float32),
        "intrinsics": intrinsics.astype(np.float32),
        "raw_cam_c2w": c2w.astype(np.float32),
    }
    if "cam_c2w_fsff" in data:
        output_data["raw_cam_c2w_fsff"] = np.asarray(data["cam_c2w_fsff"], dtype=np.float32)
    if "intrinsics_fsff" in data:
        output_data["raw_intrinsics_fsff"] = np.asarray(data["intrinsics_fsff"], dtype=np.float32)
    np.savez(output_path, **output_data)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        f"translation_sigma={args.translation_sigma}, rotation_sigma={args.rotation_sigma}, "
        f"mode={args.mode}, anchor_first={args.anchor_first}"
    )
    print_summary("raw", summarize(c2w))
    print_summary("smoothed", summarize(smoothed))


if __name__ == "__main__":
    parser = ArgumentParser(description="Gaussian-smooth a Vista4D cameras.npz C2W trajectory.")
    parser.add_argument("--input", required=True, help="Path to the input cameras.npz.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--translation_sigma", type=float, default=4.0)
    parser.add_argument("--rotation_sigma", type=float, default=4.0)
    parser.add_argument(
        "--mode",
        default="nearest",
        choices=["reflect", "constant", "nearest", "mirror", "wrap"],
        help="Boundary mode passed to scipy.ndimage.gaussian_filter1d.",
    )
    parser.add_argument(
        "--no_anchor_first",
        dest="anchor_first",
        action="store_false",
        help="Do not re-anchor the smoothed trajectory to the original first camera pose.",
    )
    parser.set_defaults(anchor_first=True)
    main(parser.parse_args())
