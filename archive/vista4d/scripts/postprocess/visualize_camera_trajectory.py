"""Plot a Vista4D camera trajectory before and after Gaussian smoothing."""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation

RAW_COLOR = "#d1495b"
SMOOTH_COLOR = "#1f77b4"
SEAM_COLOR = "#9a9a9a"
MIN_BOX_ASPECT = 0.16

# Each basis maps world XYZ onto the 3D plot's (lateral, depth, vertical) screen axes, so
# that forward travel runs horizontally and world up runs up the page. Vista4D/DA3 stitches
# into the first camera's OpenCV frame: X right, Y down, Z forward.
AXIS_CONVENTIONS = {
    "opencv": (
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
        ("X  (right)", "Z  (forward)", "-Y  (up)"),
        ("X right", "Y down", "Z forward"),
    ),
    "y_up": (
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        ("X  (right)", "Z  (forward)", "Y  (up)"),
        ("X right", "Y up", "Z forward"),
    ),
    "raw": (np.eye(3), ("X", "Y", "Z"), ("X", "Y", "Z")),
}


def fit_world_up(recon_dir: Path, c2w: np.ndarray, intrinsics: np.ndarray,
                 samples: int = 6) -> np.ndarray:
    """Estimate world up by fitting the ground plane in a few frames' depth maps."""
    import cv2
    import Imath
    import OpenEXR

    half = Imath.PixelType(Imath.PixelType.HALF)
    normals = []
    for frame in np.linspace(0, len(c2w) - 1, samples).round().astype(int):
        handle = OpenEXR.InputFile(str(recon_dir / "depths" / f"{frame:05d}.exr"))
        window = handle.header()["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        depth = np.frombuffer(handle.channel("Y", half), dtype=np.float16)
        depth = depth.reshape(height, width).astype(np.float64)
        handle.close()

        sky_path = recon_dir / "sky_mask" / f"{frame:05d}.png"
        sky = (cv2.imread(str(sky_path), cv2.IMREAD_GRAYSCALE) > 127) if sky_path.exists() \
            else np.zeros_like(depth, dtype=bool)

        fx, fy, cx, cy = intrinsics[frame]
        rows, cols = np.mgrid[0:height, 0:width]
        # The lower part of the frame is ground for a forward-facing camera; drop sky and
        # far/invalid depth so facades and distant clutter do not tilt the fit.
        keep = (rows > 0.55 * height) & np.isfinite(depth) & (depth > 0.1) & (depth < 30) & ~sky
        if keep.sum() < 1000:
            continue
        z = depth[keep]
        points = np.stack([(cols[keep] - cx) / fx * z, (rows[keep] - cy) / fy * z, z], axis=1)
        for _ in range(6):  # iteratively reweighted least squares against outliers
            center = points.mean(axis=0)
            normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
            residual = np.abs((points - center) @ normal)
            points = points[residual < max(2.5 * float(np.median(residual)), 1e-3)]
        world_normal = c2w[frame, :3, :3] @ normal
        world_normal /= np.linalg.norm(world_normal)
        # Camera +Y points down, so the up-facing ground normal has a negative Y component.
        normals.append(world_normal * (1.0 if world_normal[1] < 0 else -1.0))

    if not normals:
        raise ValueError(f"Could not fit a ground plane from {recon_dir}")
    up = np.stack(normals).mean(axis=0)
    return up / np.linalg.norm(up)


def gravity_basis(up: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Map world XYZ onto (lateral, horizontal-forward, up) given a world up vector.

    Vista4D stitches into the first camera's frame, so world -Y is only "up" when that camera
    happened to be level. Supplying the scene's real up (e.g. a ground-plane normal) re-levels
    the plot, which is what separates genuine climb from a pitched-down camera walking flat.
    """
    up = np.asarray(up, dtype=np.float64)
    norm = np.linalg.norm(up)
    if norm < 1e-8:
        raise ValueError("--up_vector must be non-zero")
    up = up / norm

    travel = c2w[-1, :3, 3] - c2w[0, :3, 3]
    forward = travel - (travel @ up) * up
    if np.linalg.norm(forward) < 1e-6:
        # Purely vertical path: fall back to the first camera's optical axis for a heading.
        forward = c2w[0, :3, 2] - (c2w[0, :3, 2] @ up) * up
    forward = forward / np.linalg.norm(forward)

    lateral = np.cross(forward, up)
    lateral = lateral / np.linalg.norm(lateral)
    if lateral @ c2w[0, :3, 0] < 0:  # keep "+lateral" on the first camera's right
        lateral = -lateral
    return np.stack([lateral, forward, up])


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def load_pair(smooth_path: Path, raw_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_c2w, smoothed_c2w) as float64 [N,4,4]."""
    data = np.load(smooth_path)
    if "cam_c2w" not in data:
        raise ValueError(f"{smooth_path} has no cam_c2w array")
    smoothed = np.asarray(data["cam_c2w"], dtype=np.float64)

    if raw_path is not None:
        raw = np.asarray(np.load(raw_path)["cam_c2w"], dtype=np.float64)
    elif "raw_cam_c2w" in data:
        raw = np.asarray(data["raw_cam_c2w"], dtype=np.float64)
    else:
        raise ValueError(
            f"{smooth_path} has no raw_cam_c2w; pass --raw pointing at the pre-smoothing cameras.npz"
        )

    if raw.shape != smoothed.shape:
        raise ValueError(f"Shape mismatch: raw {raw.shape} vs smoothed {smoothed.shape}")
    if smoothed.ndim != 3 or smoothed.shape[1:] != (4, 4):
        raise ValueError(f"Expected cam_c2w shape [N,4,4], got {smoothed.shape}")
    return raw, smoothed


def step_rotation_deg(c2w: np.ndarray) -> np.ndarray:
    """Frame-to-frame relative rotation magnitude in degrees, length N-1."""
    rel = np.einsum("nji,njk->nik", c2w[:-1, :3, :3], c2w[1:, :3, :3])
    trace = np.trace(rel, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def pairwise_rotation_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame rotation difference between two trajectories, in degrees."""
    rel = np.einsum("nji,njk->nik", a[:, :3, :3], b[:, :3, :3])
    trace = np.trace(rel, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def read_window_starts(manifest_path: Path) -> list[int]:
    manifest = json.loads(manifest_path.read_text())
    starts = [int(clip["start_frame"]) for clip in manifest.get("clips", [])]
    return [s for s in starts if s > 0]


def set_box_aspect(ax, points: np.ndarray, equal: bool) -> bool:
    """Scale the 3D box; returns True if a short axis had to be widened to stay readable."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    spans = np.maximum(hi - lo, 1e-9)
    pad = spans * 0.05
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])

    # Every axis is drawn over its own data range either way; the box aspect decides whether
    # one world unit is the same on-screen length on each axis (equal) or each axis is
    # stretched to fill the box, which exaggerates lateral wobble on a forward-dominated path.
    spans = hi - lo
    aspect = spans / spans.max() if equal else np.ones(3)
    # A near-planar path would otherwise collapse its short axis into an unreadable sliver.
    clamped = bool((aspect < MIN_BOX_ASPECT).any())
    aspect = np.maximum(aspect, MIN_BOX_ASPECT)
    ax.set_box_aspect(tuple(aspect), zoom=1.08)

    # A squeezed axis has room for far fewer tick labels than a full-length one.
    for axis, ratio in zip((ax.xaxis, ax.yaxis, ax.zaxis), aspect):
        axis.set_major_locator(
            MaxNLocator(nbins=int(np.clip(round(6 * ratio), 2, 6)), prune="both")
        )
    return clamped


def draw_orientation(ax, c2w: np.ndarray, basis: np.ndarray, stride: int, color: str,
                     length: float) -> None:
    if stride <= 0:
        return
    idx = np.arange(0, len(c2w), stride)
    origins = c2w[idx, :3, 3] @ basis.T
    # Camera +Z (viewing direction) in world coordinates; a basis swap rotates it the same way.
    forward = c2w[idx, :3, 2] @ basis.T
    ax.quiver(
        origins[:, 0], origins[:, 1], origins[:, 2],
        forward[:, 0], forward[:, 1], forward[:, 2],
        length=length, color=color, alpha=0.55, linewidth=0.8, arrow_length_ratio=0.25,
    )


def mark_seams(ax, starts: list[int], label_once: bool) -> None:
    for i, start in enumerate(starts):
        ax.axvline(
            start, color=SEAM_COLOR, linestyle=":", linewidth=0.9, zorder=0,
            label="window start" if (label_once and i == 0) else None,
        )


def summarize(raw: np.ndarray, smoothed: np.ndarray) -> dict:
    t_raw, t_sm = raw[:, :3, 3], smoothed[:, :3, 3]
    step_raw = np.linalg.norm(np.diff(t_raw, axis=0), axis=1)
    step_sm = np.linalg.norm(np.diff(t_sm, axis=0), axis=1)
    rot_raw, rot_sm = step_rotation_deg(raw), step_rotation_deg(smoothed)
    dev_t = np.linalg.norm(t_raw - t_sm, axis=1)
    dev_r = pairwise_rotation_deg(raw, smoothed)
    return {
        "frames": int(len(raw)),
        "path_length_raw": float(step_raw.sum()),
        "path_length_smoothed": float(step_sm.sum()),
        "translation_step_mean_raw": float(step_raw.mean()),
        "translation_step_mean_smoothed": float(step_sm.mean()),
        "translation_step_max_raw": float(step_raw.max()),
        "translation_step_max_smoothed": float(step_sm.max()),
        "rotation_step_mean_deg_raw": float(rot_raw.mean()),
        "rotation_step_mean_deg_smoothed": float(rot_sm.mean()),
        "rotation_step_max_deg_raw": float(rot_raw.max()),
        "rotation_step_max_deg_smoothed": float(rot_sm.max()),
        "deviation_translation_mean": float(dev_t.mean()),
        "deviation_translation_max": float(dev_t.max()),
        "deviation_rotation_mean_deg": float(dev_r.mean()),
        "deviation_rotation_max_deg": float(dev_r.max()),
    }


def print_summary(stats: dict) -> None:
    def row(name: str, raw_key: str, smooth_key: str) -> None:
        raw_value, smooth_value = stats[raw_key], stats[smooth_key]
        ratio = smooth_value / raw_value if raw_value else float("nan")
        print(f"  {name:<28s} {raw_value:>12.6f} {smooth_value:>12.6f} {ratio:>8.3f}")

    print(f"  frames: {stats['frames']}")
    print(f"  {'metric':<28s} {'raw':>12s} {'smoothed':>12s} {'ratio':>8s}")
    row("translation path length", "path_length_raw", "path_length_smoothed")
    row("translation step mean", "translation_step_mean_raw", "translation_step_mean_smoothed")
    row("translation step max", "translation_step_max_raw", "translation_step_max_smoothed")
    row("rotation step mean (deg)", "rotation_step_mean_deg_raw", "rotation_step_mean_deg_smoothed")
    row("rotation step max (deg)", "rotation_step_max_deg_raw", "rotation_step_max_deg_smoothed")
    print(
        f"  deviation translation mean/max: "
        f"{stats['deviation_translation_mean']:.6f} / {stats['deviation_translation_max']:.6f}"
    )
    print(
        f"  deviation rotation mean/max:    "
        f"{stats['deviation_rotation_mean_deg']:.6f} / {stats['deviation_rotation_max_deg']:.6f} deg"
    )


def build_figure(raw: np.ndarray, smoothed: np.ndarray, starts: list[int], args) -> plt.Figure:
    frames = np.arange(len(raw))
    t_raw, t_sm = raw[:, :3, 3], smoothed[:, :3, 3]

    fig = plt.figure(figsize=(17.0, 10.0))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

    basis, axis_labels, world_labels = AXIS_CONVENTIONS[args.axis_convention]
    if args.up_vector is not None:
        basis = gravity_basis(args.up_vector, smoothed)
        axis_labels = ("lateral  (right)", "forward  (horizontal)", "up  (gravity)")
        view_name = "gravity-levelled"
    else:
        view_name = f"{args.axis_convention} axes"
    p_raw, p_sm = t_raw @ basis.T, t_sm @ basis.T

    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax3d.plot(*p_raw.T, color=RAW_COLOR, linewidth=1.4, alpha=0.85, label="raw")
    ax3d.plot(*p_sm.T, color=SMOOTH_COLOR, linewidth=1.8, label="smoothed")
    both = np.concatenate([p_raw, p_sm], axis=0)
    arrow_length = float(np.linalg.norm(both.max(axis=0) - both.min(axis=0))) * 0.035
    draw_orientation(ax3d, raw, basis, args.orientation_stride, RAW_COLOR, arrow_length)
    draw_orientation(ax3d, smoothed, basis, args.orientation_stride, SMOOTH_COLOR, arrow_length)
    ax3d.scatter(*p_sm[0], color="#2a9d8f", s=45, depthshade=False, label="frame 0")
    ax3d.scatter(*p_sm[-1], color="#e9c46a", s=45, depthshade=False, label=f"frame {len(raw) - 1}")
    clamped = set_box_aspect(ax3d, both, args.equal_aspect)
    ax3d.set_xlabel(axis_labels[0])
    ax3d.set_ylabel(axis_labels[1])
    ax3d.set_zlabel(axis_labels[2])
    ax3d.view_init(elev=args.elev, azim=args.azim)
    if args.equal_aspect:
        scale_note = "equal aspect, short axis widened" if clamped else "equal aspect"
    else:
        scale_note = "per-axis autoscale"
    ax3d.set_title(
        f"Camera centers, {view_name} ({scale_note}; arrows = viewing direction)"
    )
    ax3d.legend(loc="upper left", fontsize=8)

    ax_axis = fig.add_subplot(grid[0, 1])
    for i, (axis_name, axis_color) in enumerate(zip(world_labels, ["#e76f51", "#2a9d8f", "#264653"])):
        ax_axis.plot(frames, t_raw[:, i] - t_raw[0, i], color=axis_color, linewidth=0.9,
                     linestyle="--", alpha=0.7, label=f"{axis_name} raw")
        ax_axis.plot(frames, t_sm[:, i] - t_sm[0, i], color=axis_color, linewidth=1.5,
                     label=f"{axis_name} smoothed")
    mark_seams(ax_axis, starts, label_once=True)
    ax_axis.set_ylabel("offset from frame 0")
    ax_axis.set_title("Per-axis translation (world coordinates, unremapped)")
    ax_axis.legend(fontsize=7, ncol=4, loc="best")

    ax_tstep = fig.add_subplot(grid[1, 1], sharex=ax_axis)
    ax_tstep.plot(frames[1:], np.linalg.norm(np.diff(t_raw, axis=0), axis=1),
                  color=RAW_COLOR, linewidth=1.0, alpha=0.85, label="raw")
    ax_tstep.plot(frames[1:], np.linalg.norm(np.diff(t_sm, axis=0), axis=1),
                  color=SMOOTH_COLOR, linewidth=1.4, label="smoothed")
    mark_seams(ax_tstep, starts, label_once=False)
    ax_tstep.set_ylabel("|Δt| per frame")
    ax_tstep.set_title("Translation step magnitude (jitter)")
    ax_tstep.legend(fontsize=7)

    ax_rstep = fig.add_subplot(grid[2, 1], sharex=ax_axis)
    ax_rstep.plot(frames[1:], step_rotation_deg(raw), color=RAW_COLOR, linewidth=1.0,
                  alpha=0.85, label="raw")
    ax_rstep.plot(frames[1:], step_rotation_deg(smoothed), color=SMOOTH_COLOR, linewidth=1.4,
                  label="smoothed")
    mark_seams(ax_rstep, starts, label_once=False)
    ax_rstep.set_ylabel("deg per frame")
    ax_rstep.set_title("Rotation step magnitude (jitter)")
    ax_rstep.legend(fontsize=7)

    ax_dev = fig.add_subplot(grid[3, 1], sharex=ax_axis)
    ax_dev.plot(frames, np.linalg.norm(t_raw - t_sm, axis=1), color="#6a4c93",
                linewidth=1.2, label="translation")
    ax_dev.set_ylabel("|raw - smoothed|")
    ax_dev.set_xlabel("frame")
    ax_dev_r = ax_dev.twinx()
    ax_dev_r.plot(frames, pairwise_rotation_deg(raw, smoothed), color="#f4a261",
                  linewidth=1.2, label="rotation")
    ax_dev_r.set_ylabel("deg")
    mark_seams(ax_dev, starts, label_once=False)
    ax_dev.set_title("How far smoothing moved each frame")
    handles = ax_dev.get_lines()[:1] + ax_dev_r.get_lines()[:1]
    ax_dev.legend(handles, [h.get_label() for h in handles], fontsize=7)

    fig.suptitle(args.title, fontsize=13)
    return fig


def main(args) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    smooth_path = resolve_path(args.smooth, repo_root)
    raw_path = resolve_path(args.raw, repo_root) if args.raw else None
    output_path = (
        resolve_path(args.output, repo_root)
        if args.output
        else smooth_path.with_name("camera_trajectory_smoothing.png")
    )

    raw, smoothed = load_pair(smooth_path, raw_path)
    starts: list[int] = []
    for manifest in args.manifest or []:
        starts.extend(read_window_starts(resolve_path(manifest, repo_root)))
    starts = sorted(set(s for s in starts if s < len(raw)))

    if args.up_from_recon:
        recon_dir = resolve_path(args.up_from_recon, repo_root)
        intrinsics = np.asarray(np.load(smooth_path)["intrinsics"], dtype=np.float64)
        args.up_vector = fit_world_up(recon_dir, smoothed, intrinsics).tolist()
        print(f"Fitted world up from {recon_dir}: {np.round(args.up_vector, 4).tolist()}")

    if not args.title:
        args.title = f"Camera trajectory before/after smoothing — {smooth_path.parent.parent.name}"

    fig = build_figure(raw, smoothed, starts, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    stats = summarize(raw, smoothed)
    if args.up_vector is not None:
        basis = gravity_basis(args.up_vector, smoothed)
        travel = smoothed[-1, :3, 3] - smoothed[0, :3, 3]
        length = float(np.linalg.norm(travel))
        vertical = float(travel @ basis[2])
        # Elevation of the optical axis above the ground plane: negative means looking down.
        pitch = float(np.degrees(np.arcsin(np.clip(smoothed[0, :3, 2] @ basis[2], -1.0, 1.0))))
        stats.update({
            "travel_length": length,
            "travel_vertical": vertical,
            "travel_horizontal": float(np.linalg.norm(travel - vertical * basis[2])),
            "travel_tilt_vs_ground_deg": float(np.degrees(np.arcsin(abs(vertical) / length))),
            "first_camera_pitch_deg": pitch,
        })

    print(f"Smoothed: {smooth_path}")
    print(f"Raw:      {raw_path if raw_path else f'{smooth_path} (raw_cam_c2w)'}")
    if starts:
        print(f"Window starts: {starts}")
    print_summary(stats)
    if args.up_vector is not None:
        print(
            f"  travel {stats['travel_length']:.3f} = horizontal {stats['travel_horizontal']:.3f}"
            f" + vertical {stats['travel_vertical']:+.3f}"
            f"  (tilt vs ground {stats['travel_tilt_vs_ground_deg']:.1f} deg)"
        )
        print(
            f"  first camera pitch vs ground: {stats['first_camera_pitch_deg']:+.1f} deg"
            f" ({'looking down' if stats['first_camera_pitch_deg'] < 0 else 'looking up'})"
        )
    print(f"Figure: {output_path}")

    if args.stats_json:
        stats_path = resolve_path(args.stats_json, repo_root)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"Stats:  {stats_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Visualize a Vista4D camera trajectory before and after Gaussian smoothing."
    )
    parser.add_argument(
        "--smooth",
        required=True,
        help="Path to cameras_gaussian_smooth.npz (uses its embedded raw_cam_c2w unless --raw is given).",
    )
    parser.add_argument("--raw", default=None, help="Optional explicit pre-smoothing cameras.npz.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="Splits manifest JSON; window start frames are drawn as vertical guides. Repeatable.",
    )
    parser.add_argument("--orientation_stride", type=int, default=10,
                        help="Draw a viewing-direction arrow every N frames (0 disables).")
    parser.add_argument(
        "--axis_convention",
        default="opencv",
        choices=sorted(AXIS_CONVENTIONS),
        help="How to orient the 3D plot. 'opencv' (default) assumes the Vista4D/DA3 world frame "
             "(X right, Y down, Z forward) and draws forward travel horizontally with world up "
             "up the page; 'y_up' for a Y-up world; 'raw' plots world XYZ unchanged.",
    )
    parser.add_argument(
        "--up_vector",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="World up vector (e.g. a ground-plane normal). Overrides --axis_convention and "
             "re-levels the 3D plot onto (lateral, horizontal forward, up), so a pitched-down "
             "camera walking on flat ground no longer reads as a climb.",
    )
    parser.add_argument(
        "--up_from_recon",
        default=None,
        help="Fit world up from a recon_and_seg directory's depths/ and sky_mask/ instead of "
             "passing --up_vector by hand.",
    )
    parser.add_argument(
        "--no_equal_aspect",
        dest="equal_aspect",
        action="store_false",
        help="Autoscale each 3D axis independently instead of keeping a geometrically faithful "
             "1:1:1 box. Useful when travel along one axis dwarfs the lateral wobble.",
    )
    parser.set_defaults(equal_aspect=True)
    # Near-side elevation: the forward axis runs across the page, world up runs up it.
    parser.add_argument("--elev", type=float, default=16.0)
    parser.add_argument("--azim", type=float, default=-24.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--title", default=None)
    parser.add_argument("--stats_json", default=None, help="Optional path to dump metrics as JSON.")
    main(parser.parse_args())
