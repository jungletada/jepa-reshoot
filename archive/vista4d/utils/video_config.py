"""Strict, CPU-only configuration for the video workflow (paths relative to repo)."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

import yaml

from utils.video_experiment import ROOT, DIMENSIONS

PROMPT = ('A realistic handheld smartphone video of people in an everyday scene, '
          'with natural body motion, realistic lighting, stable camera motion, '
          'and detailed surroundings.')
DEFAULTS = {
    'schema_version': 1,
    'video': {'path': '', 'resolution': '384p', 'mode': 'full'},
    'single_clip': {'start_frame': 0, 'name': None},
    'preprocess': {'reconstruction': 'da3', 'da3_process_res': 'auto',
                   'segmentation_keywords': ['person', 'man', 'woman', 'hand', 'phone',
                                             'bag', 'backpack', 'car', 'stroller'],
                   'save_visuals': False},
    'camera': {'use_smoothed': True, 'translation_sigma': 8.0, 'rotation_sigma': 10.0},
    'render': {'mode': 'auto', 'static_frame_stride': 4, 'chunk_size': 4, 'save_visuals': False},
    'windows': {'frames': 49, 'baseline_overlap': 5, 'flowlong_overlap': 25,
                'temporal_alignment': 4, 'tail_padding': 'repeat_last'},
    'inference': {'seed': 10027, 'steps': 50, 'cfg_scale': 5.0, 'sigma_shift': 5.0,
                  'tile_vae': True, 'microbatch_size': 1, 'prompt': PROMPT},
    'flowlong': {'baseline': True, 'matching_only': True, 'stochastic_thresholds': [0.5, 0.6]},
    'models': {'vista4d': 'auto'},
    'outputs': {'run_name': 'default'},
}


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in mapping:
            raise ValueError(f'YAML keys must be unique strings: {key!r}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def merge(defaults, supplied, prefix=''):
    if not isinstance(supplied, dict):
        raise ValueError(f'{prefix or "config"} must be a mapping')
    result = deepcopy(defaults)
    for key, value in supplied.items():
        name = f'{prefix}.{key}' if prefix else key
        if key not in defaults:
            raise ValueError(f'Unknown configuration key: {name}')
        result[key] = merge(defaults[key], value, name) if isinstance(defaults[key], dict) else value
    return result


def threshold_name(value):
    text = format(Decimal(str(float(value))), 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return 'flowlong_t' + text.replace('.', 'p')


def resolve_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate(config):
    def number(value, name, minimum=0, integer=False, maximum=None):
        if (isinstance(value, bool) or not isinstance(value, int if integer else (int, float))
                or not math.isfinite(value) or value < minimum
                or (maximum is not None and value > maximum)):
            raise ValueError(f'Invalid {name}: {value!r}')

    # Scalar types, including booleans, are checked before interpreting options.
    for section, defaults in DEFAULTS.items():
        if not isinstance(defaults, dict):
            continue
        for key, default in defaults.items():
            value = config[section][key]
            name = f'{section}.{key}'
            if isinstance(default, bool) and type(value) is not bool:
                raise ValueError(f'{name} must be a YAML boolean, not a string')
            if isinstance(default, str) and not isinstance(value, str) and name != 'preprocess.da3_process_res':
                raise ValueError(f'{name} must be a string')
            if isinstance(default, list) and not isinstance(value, list):
                raise ValueError(f'{name} must be a list')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('Only schema_version: 1 is supported')
    video = config['video']
    if not video['path'].strip() or video['resolution'] not in DIMENSIONS or video['mode'] not in ('full', 'single_clip'):
        raise ValueError('Specify video.path, resolution (384p/720p), and mode (full/single_clip)')
    for name in (Path(video['path']).stem, config['outputs']['run_name']):
        if not re.fullmatch(r'[\w-]+', name):
            raise ValueError(f'Unsafe video/run name: {name!r}')
    clip = config['single_clip']
    number(clip['start_frame'], 'single_clip.start_frame', integer=True)
    if clip['name'] is not None and (not isinstance(clip['name'], str) or not re.fullmatch(r'[\w-]+', clip['name'])):
        raise ValueError('single_clip.name must be null or a simple filename stem')
    if video['mode'] == 'full' and (clip['start_frame'] != 0 or clip['name'] is not None):
        raise ValueError('single_clip overrides apply only to video.mode: single_clip')
    if config['windows'] != DEFAULTS['windows'] or any(type(config['windows'][k]) is not type(v) for k,v in DEFAULTS['windows'].items()):
        raise ValueError('Supported windows: 49 frames, baseline overlap=5, FlowLong overlap=25, alignment=4, repeat_last')
    pre = config['preprocess']
    if pre['reconstruction'] != 'da3':
        raise ValueError('Configured workflow currently supports reconstruction: da3')
    if pre['da3_process_res'] != 'auto':
        number(pre['da3_process_res'], 'preprocess.da3_process_res', 1, integer=True)
    if not pre['segmentation_keywords'] or any(not isinstance(s, str) or not s.strip() or any(c.isspace() for c in s) for s in pre['segmentation_keywords']):
        raise ValueError('segmentation_keywords must be a nonempty list of single tokens')
    for key in ('translation_sigma', 'rotation_sigma'):
        number(config['camera'][key], f'camera.{key}')
    if video['mode'] == 'full' and not config['camera']['use_smoothed']:
        raise ValueError('Full shared-static workflow requires camera.use_smoothed: true')
    expected_render = 'shared_static' if video['mode'] == 'full' else 'single'
    if config['render']['mode'] not in ('auto', expected_render):
        raise ValueError(f'This video mode requires render.mode: {expected_render} (or auto)')
    config['render']['mode'] = expected_render
    for key in ('static_frame_stride', 'chunk_size'):
        number(config['render'][key], f'render.{key}', 1, integer=True)
    inf = config['inference']
    for key in ('steps', 'microbatch_size'):
        number(inf[key], f'inference.{key}', 1, integer=True)
    number(inf['seed'], 'inference.seed', integer=True)
    number(inf['cfg_scale'], 'inference.cfg_scale', 0)
    number(inf['sigma_shift'], 'inference.sigma_shift', 1e-9)
    if not inf['prompt'].strip():
        raise ValueError('inference.prompt cannot be empty')
    thresholds = config['flowlong']['stochastic_thresholds']
    for value in thresholds:
        number(value, 'stochastic_threshold', maximum=1)
        if value == 0:
            raise ValueError('stochastic_threshold must satisfy 0 < threshold <= 1; use matching_only to disable stochastic sampling')
    if len({threshold_name(t) for t in thresholds}) != len(thresholds):
        raise ValueError('Duplicate stochastic thresholds')
    if not config['models']['vista4d'].strip():
        raise ValueError('models.vista4d must be auto or a checkpoint directory')
    return config


def load_config(path, overrides=None):
    with resolve_path(path).open() as handle:
        supplied = yaml.load(handle, Loader=UniqueLoader)
    config = merge(DEFAULTS, supplied)
    for section, fields in (overrides or {}).items():
        config[section].update(fields)
    validate(config)
    config['video']['path'] = str(resolve_path(config['video']['path']))
    if config['models']['vista4d'] != 'auto':
        config['models']['vista4d'] = str(resolve_path(config['models']['vista4d']))
    return config


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
