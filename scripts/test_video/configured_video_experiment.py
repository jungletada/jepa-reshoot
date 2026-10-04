"""YAML front end; legacy shell scripts remain implementation details."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

from utils.video_config import load_config, fingerprint, file_sha256
from utils.video_experiment import ROOT, DIMENSIONS, probe_source, window_layout

FULL_STAGES = ('split', 'recon', 'stitch', 'render', 'inference')
SINGLE_STAGES = ('split', 'recon', 'smooth', 'render', 'inference')


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--video')
    p.add_argument('--resolution', choices=DIMENSIONS)
    p.add_argument('--seed', type=int)
    p.add_argument('--prompt')
    p.add_argument('--seg-keywords')
    p.add_argument('--vista4d-folder')
    p.add_argument('--run-name')
    p.add_argument('--mode', choices=('full', 'single_clip'))
    p.add_argument('--steps', type=int)
    p.add_argument('--stages', nargs='+', choices=(*FULL_STAGES, 'smooth'))
    p.add_argument('--gpu')
    p.add_argument('--execute', action='store_true')
    return p


def configuration(args):
    mapping = {'video': ('video','path'), 'resolution': ('video','resolution'),
               'mode': ('video','mode'), 'seed': ('inference','seed'),
               'steps': ('inference','steps'), 'prompt': ('inference','prompt'),
               'vista4d_folder': ('models','vista4d'), 'run_name': ('outputs','run_name')}
    overrides = {}
    for arg, (section, key) in mapping.items():
        value = getattr(args, arg)
        if value is not None:
            overrides.setdefault(section, {})[key] = value
    if args.seg_keywords is not None:
        overrides.setdefault('preprocess', {})['segmentation_keywords'] = args.seg_keywords.split()
    return load_config(args.config, overrides)


def make_plan(config, source, stages=None):
    c = config
    single = c['video']['mode'] == 'single_clip'
    order = SINGLE_STAGES if single else FULL_STAGES
    stages = list(stages if stages is not None else (SINGLE_STAGES if single else FULL_STAGES[:5]))
    if not stages or stages != [s for s in order if s in stages]:
        raise ValueError(f'Stages must be unique and ordered: {order}')
    target, res = Path(source['path']).stem, c['video']['resolution']
    width, height = DIMENSIONS[res]
    root = ROOT / 'results/configured' / target / res / c['outputs']['run_name']
    paths = {k: str(root / v) for k,v in {
        'root': '.', 'baseline_splits': 'media/baseline', 'flowlong_splits': 'media/flowlong',
        'single_media': 'media/single', 'reconstruction': 'reconstruction/single',
        'stitched': 'reconstruction/stitched', 'full': 'full',
        'baseline_conditions': 'conditions/baseline', 'flowlong_conditions': 'conditions/flowlong',
        'inference': 'inference', 'logs': 'logs',
    }.items()}
    paths['full_sequence'] = str(Path(paths['full']) / f'{target}_stitched_{res}')
    phases = []
    if c['flowlong']['baseline']:
        phases.append('baseline')
    if c['flowlong']['matching_only']:
        phases.append('matching_only')
    phases += ['t' + str(float(t)) for t in c['flowlong']['stochastic_thresholds']]
    if not single and 'inference' in stages and not phases:
        raise ValueError('No inference variants selected')
    layouts = {'k1': window_layout(source['frames'])}
    if (not single and 'inference' in stages and any(p != 'baseline' for p in phases)
            and not layouts['k1']['flowlong_supported']):
        raise ValueError('FlowLong requires at least two 49-frame windows')
    start = c['single_clip']['start_frame']
    clip_name = c['single_clip']['name'] or f'{target}_frames{start:06d}_{start+48:06d}_{res}49'
    if single and start + 49 > source['frames']:
        raise ValueError('Selected single clip must contain 49 source frames')
    return dict(config=c, source=source, target=target, resolution=res, width=width, height=height,
                paths=paths, stages=stages, phases=phases, layouts=layouts,
                clip_name=clip_name)


def jobs(plan):
    c, p = plan['config'], plan['paths']
    target, res = plan['target'], plan['resolution']
    inf, pre, camera = c['inference'], c['preprocess'], c['camera']
    single = c['video']['mode'] == 'single_clip'
    checkpoint = c['models']['vista4d']
    if checkpoint == 'auto':
        checkpoint = str(ROOT/'checkpoints/vista4d'/('384p49_step=30000' if res=='384p' else '720p49_step=3000'))
    env = dict(RESOLUTION=res, WIDTH=plan['width'], HEIGHT=plan['height'],
               NUM_FRAMES=49, CLIP_FRAMES=49, TEMPORAL_ALIGNMENT=4, INCLUDE_TAIL=True,
               SEED=inf['seed'], SEEDS=inf['seed'], NUM_INFERENCE_STEPS=inf['steps'],
               CFG_SCALE=inf['cfg_scale'], SIGMA_SHIFT=inf['sigma_shift'], PROMPT=inf['prompt'],
               TILE_VAE=inf['tile_vae'], FLOWLONG_MICROBATCH_SIZE=inf['microbatch_size'],
               VISTA4D_FOLDER=checkpoint, FORCE=False, DRY_RUN=False, USE_USP=False, CFG_MERGE=False,
               FLOWLONG_DISABLE_STOCHASTIC=False,
               RECON_METHOD=pre['reconstruction'], SEG_KEYWORDS=' '.join(pre['segmentation_keywords']),
               DA3_PROCESS_RES=plan['width'] if pre['da3_process_res']=='auto' else pre['da3_process_res'],
               TRANSLATION_SIGMA=camera['translation_sigma'], ROTATION_SIGMA=camera['rotation_sigma'],
               USE_SMOOTHED_CAMERA=camera['use_smoothed'],
               STATIC_FRAME_STRIDE=c['render']['static_frame_stride'], RENDER_CHUNK_SIZE=c['render']['chunk_size'],
               BASELINE_SPLITS_DIR=p['baseline_splits'], FLOWLONG_SPLITS_DIR=p['flowlong_splits'],
               FULL_RESULT_BASE=p['full'], FULL_SEQUENCE_ROOT=p['full_sequence'],
               FULL_RECON_FOLDER=str(Path(p['full_sequence'])/'recon_and_seg'),
               BASELINE_CONDITION_ROOT=p['baseline_conditions'], FLOWLONG_RESULT_ROOT=p['flowlong_conditions'],
               EVAL_ROOT=p['inference'], LOG_ROOT=p['logs'],
               ALLOW_CUSTOM_INFERENCE=True)
    if single:
        result = str(Path(p['reconstruction'])/plan['clip_name'])
        env.update(SOURCE_VIDEO=str(Path(p['single_media'])/(plan['clip_name']+'.mp4')),
                   EXAMPLE=plan['clip_name'], CLIP_NAME=plan['clip_name'], RESULT_ROOT=result)
    def strings(values):
        return {k: str(v).lower() if isinstance(v,bool) else str(v) for k,v in values.items()}
    output = []
    def add(stage, command, **extra):
        output.append(dict(stage=stage, command=command, env=strings({**env, **extra})))
    for stage in plan['stages']:
        if stage == 'split':
            if single:
                add(stage,[sys.executable,'-m','scripts.preprocess.prepare_custom_single_video',
                           '--input',plan['source']['path'],'--output_dir',p['single_media'],
                           '--output_name',plan['clip_name'],'--start_frame',str(c['single_clip']['start_frame']),
                           '--num_frames','49','--width',str(plan['width']),'--height',str(plan['height'])])
            else:
                for folder, overlap in ((p['baseline_splits'],5),(p['flowlong_splits'],25)):
                    add(stage,['bash','scripts/test_video/split_video.sh',plan['source']['path']],
                        OUTPUT_DIR=folder,OVERLAP=overlap,START_FRAME=0,MAX_FRAMES=plan['source']['frames'])
        elif stage == 'recon':
            add(stage,['bash','scripts/test_video/recon_and_seg.sh'], SAVE_VIS=pre['save_visuals'])
        elif stage == 'stitch':
            add(stage,['bash','scripts/test_video/stitch_splits_smooth_and_slice.sh',target],
                SPLITS_DIR=p['baseline_splits'], INPUT_RESULT_ROOT=p['reconstruction'],
                OUTPUT_RESULT_ROOT=p['stitched'], FORCE_STITCH=False, OVERWRITE_SPLITS=False)
        elif stage == 'smooth':
            add(stage,['bash','scripts/test_video/smooth.sh'])
        elif stage == 'render':
            script = 'render.sh' if single else 'prepare_flowlong_ab_conditions.sh'
            add(stage,['bash',f'scripts/test_video/{script}',*([] if single else [target])],
                SAVE_VIS=c['render']['save_visuals'],
                OVERWRITE_FULL_RENDER=False, OVERWRITE_BASELINE_SPLITS=False, OVERWRITE_FLOWLONG_SPLITS=False)
        elif stage == 'inference':
            if single:
                add(stage,['bash','scripts/test_video/inference.sh'])
            else:
                add(stage,['bash','scripts/test_video/run_flowlong_stage4.sh',target],
                    PHASES=' '.join(plan['phases']))
    return output


def snapshot(plan):
    """No adoption/overwrite of legacy results; freeze configuration and source."""
    root = Path(plan['paths']['root'])
    effective = jobs(plan)[0]['env']
    expected = dict(config=plan['config'], source=plan['source'],
                    source_sha256=file_sha256(plan['source']['path']), paths=plan['paths'],
                    derived=dict(target=plan['target'], width=plan['width'], height=plan['height'],
                                 clip_name=plan['clip_name'], layouts=plan['layouts'],
                                 vista4d_folder=effective['VISTA4D_FOLDER'],
                                 da3_process_res=int(effective['DA3_PROCESS_RES'])))
    record = root/'resolved_config.yaml'
    if record.exists():
        actual = yaml.safe_load(record.read_text())
        if actual != expected:
            raise ValueError(f'Configuration/source changed. Choose a new outputs.run_name; refusing reuse: {root}')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f'Untracked results exist; choose a new outputs.run_name: {root}')
        root.mkdir(parents=True, exist_ok=True)
        # Exclusive create also catches accidental simultaneous first starts.
        with record.open('x') as handle:
            yaml.safe_dump(expected,handle,sort_keys=False,allow_unicode=True)


def product_stats(root):
    """Cheap resume guard, not a cryptographic verification of every tensor."""
    return {str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in root.rglob('*') if path.is_file()
            and path.relative_to(root).parts[0] not in ('logs','.state')
            and path.name not in ('resolved_config.yaml','.run.lock')}


def configured_environment():
    """Remove shell overrides while preserving the runtime environment."""
    from scripts.test_video.run_video_experiment import stage_environment
    names = set()
    for script in (ROOT/'scripts/test_video').glob('*.sh'):
        names.update(re.findall(r'\$\{([A-Z][A-Z0-9_]*)',script.read_text()))
    keep = {'PATH','HOME','LD_LIBRARY_PATH','PYTHONPATH','CONDA_PREFIX','CONDA_DEFAULT_ENV'}
    env = stage_environment()
    return {key:value for key,value in env.items()
            if (key in keep or key.startswith(('CUDA_','NCCL_','HF_','HUGGINGFACE_','TORCH_'))
                or key not in names)}


def run(args):
    from scripts.test_video.run_video_experiment import gpu_environment, validate_reconstruction
    c = configuration(args)
    source = probe_source(Path(c['video']['path']))
    plan = make_plan(c,source,args.stages)
    commands = jobs(plan)
    print(json.dumps({**plan,'commands':commands},indent=2,ensure_ascii=False),flush=True)
    if not args.execute:
        return plan
    gpu_stages = {'recon','render','inference'}
    gpu_env = gpu_environment(args.gpu) if set(plan['stages']) & gpu_stages else {}
    root = Path(plan['paths']['root'])
    snapshot(plan)
    # Advisory local lock prevents concurrent writes to the same configured run.
    import fcntl
    with (root/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is executing this run: {root}') from exc
        snapshot(plan)
        state = root/'.state'
        state.mkdir(exist_ok=True)
        now = product_stats(root)
        for marker in state.glob('*.json'):
            old = json.loads(marker.read_text())
            if any(now.get(k)!=v for k,v in old['products'].items()):
                raise ValueError(f'Changed/missing outputs from {marker.stem}; refusing downstream reuse')
        for stage in plan['stages']:
            stage_jobs = [job for job in commands if job['stage']==stage]
            signature = fingerprint(stage_jobs)
            marker = state/f'{stage}.json'
            if marker.exists():
                old = json.loads(marker.read_text())
                now = product_stats(root)
                if old['signature'] != signature or any(now.get(k)!=v for k,v in old['products'].items()):
                    raise ValueError(f'Changed/missing outputs for {stage}; inspect or use a new run_name')
                print(f'STAGE_REUSE {stage}',flush=True)
                continue
            before = product_stats(root)
            for job in stage_jobs:
                print(f'STAGE_START {stage}',flush=True)
                env = {**configured_environment(),**job['env'],**gpu_env}
                if stage=='recon' and c['video']['mode']=='full':
                    manifest = Path(plan['paths']['baseline_splits'])/f"{plan['target']}_splits_manifest.json"
                    for clip in json.loads(manifest.read_text())['clips']:
                        result = Path(plan['paths']['reconstruction'])/Path(clip['output_path']).stem
                        if result.exists():
                            raise FileExistsError(f'Partial reconstruction exists: {result}; inspect it before retrying')
                        subprocess.run(job['command'],cwd=ROOT,env={**env,'SOURCE_VIDEO':clip['output_path'],
                                       'RESULT_ROOT':str(result)},check=True)
                        validate_reconstruction(result/'recon_and_seg',49,plan['width'],plan['height'])
                else:
                    subprocess.run(job['command'],cwd=ROOT,env=env,check=True)
            after = product_stats(root)
            changed = {k:v for k,v in after.items() if before.get(k)!=v}
            marker.write_text(json.dumps(dict(signature=signature,products=changed),indent=2))
            print(f'STAGE_COMPLETE {stage}',flush=True)
    return plan


if __name__ == '__main__':
    run(parser().parse_args())
