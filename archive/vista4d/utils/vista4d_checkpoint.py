"""Vista4D checkpoint IO. Torch is imported lazily; resolve/hash never load tensors.

Legacy single-file fingerprints remain the SHA-256 of that file. Indexed packages
are identified by the index AND all referenced shards (not by a claimed source hash).
"""
from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    shards: tuple
    weight_map: dict | None = None

    @property
    def files(self):
        return (self.path, *self.shards) if self.weight_map is not None else self.shards

    @property
    def size_bytes(self):
        return sum(p.stat().st_size for p in self.files)


def resolve_checkpoint(path):
    """Resolve an explicit file or unambiguous directory; never glob-load shards."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob('*.safetensors.index.json'))
        candidates += [p for p in (path/'dit.pth', path/'model.safetensors', path/'dit.safetensors') if p.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one checkpoint entry in {path}, found {len(candidates)}; specify a file/index explicitly")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name.endswith('.safetensors.index.json'):
        data = json.loads(path.read_text(), object_pairs_hook=_unique_object)
        if not isinstance(data, dict):
            raise ValueError(f'Checkpoint index must be a JSON object: {path}')
        mapping = data.get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing/nonempty weight_map required: {path}")
        for key, name in mapping.items():
            if not isinstance(key, str) or not key or not isinstance(name, str):
                raise ValueError('weight_map requires nonempty tensor names and shard filenames')
            # Flat standard packages only; reject traversal, absolute names and symlink escapes.
            if '/' in name or '\\' in name or not name.endswith('.safetensors'):
                raise ValueError(f"Unsafe shard filename: {name}")
            shard = path.parent / name
            if shard.resolve().parent != path.parent or not shard.is_file():
                raise ValueError(f"Missing or unsafe shard: {shard}")
        shards = tuple(path.parent/name for name in sorted(set(mapping.values())))
        return Checkpoint(path, shards, mapping)
    if path.suffix not in ('.pth', '.pt', '.bin', '.safetensors'):
        raise ValueError(f"Unsupported checkpoint: {path}")
    if path.suffix == '.safetensors' and '-of-' in path.stem:
        raise ValueError('Pass the shard index/directory, not an individual numbered shard')
    return Checkpoint(path, (path,))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256(path):
    spec = resolve_checkpoint(path)
    if spec.weight_map is None:
        return file_sha256(spec.path)
    entries = [(p.name, p.stat().st_size, file_sha256(p)) for p in spec.files]
    payload = json.dumps({'format': 'vista4d-sharded-v1', 'files': entries}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def checkpoint_size(path):
    return resolve_checkpoint(path).size_bytes


def load_pth(path):
    import torch
    try:
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as exc:
        if 'mmap can only be used' not in str(exc):
            raise
        state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError('Vista4D checkpoint must be a nonempty flat tensor state_dict')
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Vista4D checkpoint must be a flat tensor state_dict (no optimizer/metadata entries)')
    return state


def safetensors_shapes(spec):
    """Check every header against the index before touching model parameters."""
    from safetensors import safe_open
    shapes = {}
    for shard in spec.shards:
        with safe_open(shard, framework='pt', device='cpu') as handle:
            keys = set(handle.keys())
            if not keys:
                raise ValueError(f'Empty shard: {shard}')
            if spec.weight_map is not None:
                expected = {k for k, name in spec.weight_map.items() if name == shard.name}
                if keys != expected:
                    raise ValueError(f'Shard/index keys mismatch: {shard}; missing={expected-keys}, extra={keys-expected}')
            if shapes.keys() & keys:
                raise ValueError(f'Duplicate tensors across shards: {shard}')
            shapes.update({key: tuple(handle.get_slice(key).get_shape()) for key in keys})
    return shapes


def load_checkpoint_into_model(model, path, key_map=None):
    """Copy shards one at a time, preserving target dtype/device and partial overlays.

    A Vista4D checkpoint covers only a subset of the base Wan model. Missing base
    keys are allowed, but unexpected/duplicate/misshaped keys are always rejected.
    """
    spec = resolve_checkpoint(path)
    key_map = key_map or {}
    pth = spec.path.suffix in ('.pth', '.pt', '.bin')
    state = load_pth(spec.path) if pth else None
    shapes = {k: tuple(v.shape) for k, v in state.items()} if pth else safetensors_shapes(spec)
    names = {key: key_map.get(key, key) for key in shapes}
    if len(set(names.values())) != len(names):
        raise ValueError('Checkpoint key mapping collision')
    target = model.state_dict()
    unexpected = set(names.values()) - target.keys()
    if unexpected:
        raise ValueError(f'Unexpected Vista4D checkpoint keys: {sorted(unexpected)}')
    for key, name in names.items():
        if shapes[key] != tuple(target[name].shape):
            raise ValueError(f'Shape mismatch for {key}: {shapes[key]} != {tuple(target[name].shape)}')
    missing = sorted(target.keys() - set(names.values()))
    del target
    if pth:
        model.load_state_dict({names[k]: v for k, v in state.items()}, strict=False)
    else:
        from safetensors.torch import load_file
        for shard in spec.shards:
            state = load_file(str(shard), device='cpu')
            model.load_state_dict({names[k]: v for k, v in state.items()}, strict=False)
            del state
    return missing, []


def wrapped_key_map(model):
    """Flatten nested offload wrappers, including consecutive .module. segments."""
    mapping = {}
    for name in model.state_dict():
        flat = name
        while '.module.' in flat:
            flat = flat.replace('.module.', '.')
        if flat in mapping:
            raise ValueError(f'Ambiguous wrapped parameter name: {flat}')
        mapping[flat] = name
    return mapping


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('resolve', 'sha256', 'info'))
    parser.add_argument('path')
    args = parser.parse_args()
    spec = resolve_checkpoint(args.path)
    if args.action == 'resolve':
        print(spec.path)
    elif args.action == 'sha256':
        print(checkpoint_sha256(spec.path))
    else:
        print(json.dumps({'path': str(spec.path), 'size_bytes': spec.size_bytes,
                          'files': [str(p) for p in spec.files],
                          'sha256': checkpoint_sha256(spec.path)}, indent=2))


if __name__ == '__main__':
    main()
