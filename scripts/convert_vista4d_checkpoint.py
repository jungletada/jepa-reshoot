"""Losslessly export a flat Vista4D .pth into indexed safetensors (CPU only)."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
import shutil
import tempfile

from utils.vista4d_checkpoint import file_sha256, load_pth, resolve_checkpoint, safetensors_shapes


def parse_size(value):
    match = re.fullmatch(r'(\d+)(B|KB|MB|GB|KiB|MiB|GiB)?', str(value))
    if not match:
        raise ValueError('Use positive bytes or an integer with B/KB/MB/GB/KiB/MiB/GiB')
    factors = {None: 1, 'B': 1, 'KB': 10**3, 'MB': 10**6, 'GB': 10**9,
               'KiB': 2**10, 'MiB': 2**20, 'GiB': 2**30}
    size = int(match[1]) * factors[match[2]]
    if size <= 0:
        raise ValueError('Shard size must be positive')
    return size


def verify_conversion(source_state, output):
    """Compare raw bytes, including NaN payloads and signed zero, without casting."""
    import torch
    from safetensors import safe_open
    spec = resolve_checkpoint(output)
    shapes = safetensors_shapes(spec)
    if set(shapes) != set(source_state):
        raise ValueError('Converted tensor names differ from source')
    for shard in spec.shards:
        with safe_open(shard, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                actual, expected = handle.get_tensor(key), source_state[key]
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise ValueError(f'Converted shape/dtype mismatch: {key}')
                if not torch.equal(actual.reshape(-1).view(torch.uint8),
                                   expected.contiguous().reshape(-1).view(torch.uint8)):
                    raise ValueError(f'Converted tensor bytes differ: {key}')


def convert_checkpoint(source, output, max_shard_size='4GB', config=None):
    import torch
    from safetensors.torch import save_file
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.suffix != '.pth' or not source.is_file():
        raise ValueError('Source must be an existing .pth file')
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing directory: {output}')
    config = Path(config).resolve() if config else source.parent/'config.yaml'
    if not config.is_file():
        raise FileNotFoundError(config)
    limit = parse_size(max_shard_size)
    state = load_pth(source)
    groups, group, size, total = [], [], 0, 0
    for key in sorted(state):
        tensor = state[key]
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f'Only dense, non-quantized tensors supported: {key}')
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > limit:
            raise ValueError(f'Tensor {key} exceeds shard limit; increase --max-shard-size')
        if group and size + nbytes > limit:
            groups.append(group)
            group, size = [], 0
        group.append(key)
        size += nbytes
        total += nbytes
    if group:
        groups.append(group)
    # Only our own fresh staging directory is cleaned on failure. Never remove user outputs.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.tmp-', dir=output.parent))
    try:
        mapping = {}
        for i, keys in enumerate(groups, 1):
            name = f'model-{i:05d}-of-{len(groups):05d}.safetensors'
            # Clone independently to preserve ALL keys, including aliased tensors.
            shard = {key: state[key].detach().cpu().contiguous().clone() for key in keys}
            save_file(shard, str(staging/name), metadata={'format': 'pt'})
            del shard
            mapping.update({key: name for key in keys})
        index = {'metadata': {'total_size': total}, 'weight_map': mapping}
        (staging/'model.safetensors.index.json').write_text(json.dumps(index, indent=2)+'\n')
        shutil.copyfile(config, staging/'config.yaml')
        verify_conversion(state, staging)
        report = {'schema_version': 1, 'source': str(source), 'source_sha256': file_sha256(source),
                  'tensor_count': len(state), 'tensor_bytes': total, 'max_shard_tensor_bytes': limit,
                  'shards': len(groups), 'dtype_policy': 'preserve', 'byte_exact_verified': True,
                  'files': {p.name: file_sha256(p) for p in sorted(staging.iterdir()) if p.is_file()}}
        (staging/'conversion_report.json').write_text(json.dumps(report, indent=2)+'\n')
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Default: config.yaml next to source')
    parser.add_argument('--max-shard-size', default='4GB', help='Tensor payload limit; safetensors headers add a small overhead')
    args = parser.parse_args()
    print(json.dumps(convert_checkpoint(args.input, args.output, args.max_shard_size, args.config), indent=2))


if __name__ == '__main__':
    main()
