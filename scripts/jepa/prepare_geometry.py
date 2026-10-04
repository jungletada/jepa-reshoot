"""Export source-only predictor geometry from the existing rendered clip folder."""
from argparse import ArgumentParser
from pathlib import Path

import torch

from jepa.features import FeatureCache
from jepa.predictor import prepare_geometry


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--render-folder", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Source feature cache for grid/time alignment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from utils.media import load_cameras, load_depths, load_masks
    folder = args.render_folder
    source_c2w, source_intrinsics = load_cameras(str(folder / "cameras_src.npz"))
    target_c2w, target_intrinsics = load_cameras(str(folder / "cameras_tgt.npz"))
    geometry = {
        "source_depth": load_depths(str(folder / "depths_src")),
        "source_c2w": source_c2w, "source_intrinsics": source_intrinsics,
        "target_c2w": target_c2w, "target_intrinsics": target_intrinsics,
        "target_visibility": load_masks(str(folder / "alpha_mask_pc")),
    }
    geometry = {k: torch.from_numpy(v.copy()).unsqueeze(0) for k, v in geometry.items()}
    source = FeatureCache.load(args.source)
    if source.role != "source":
        raise ValueError("Geometry export requires a source feature cache")
    prepare_geometry(source, geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(geometry, args.output)


if __name__ == "__main__":
    main()
