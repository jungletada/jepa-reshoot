"""Unified multi-video workflow. Planning is the default; --execute is required."""
from argparse import ArgumentParser
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from utils.video_experiment import ROOT, DIMENSIONS, probe_source, split_directory, window_layout

STAGES = ("split", "recon", "stitch", "render", "inference")
PHASES = ("baseline", "matching_only", "t0.5", "t0.6", "t0.7")


def gpu_environment(selection=None):
    """Check selected GPUs and return a UUID-based environment for all stages.

    --gpu indices are nvidia-smi physical indices, not CUDA logical indices.
    Inherited numeric CUDA ordinals require PCI_BUS_ID ordering so we can map
    them without initializing CUDA (and becoming a compute process ourselves).
    """
    def query(kind, fields):
        output = subprocess.check_output([
            'nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'
        ], text=True)
        rows = [[v.strip() for v in row] for row in csv.reader(io.StringIO(output)) if row]
        if any(len(row) != len(fields.split(',')) for row in rows):
            raise RuntimeError(f'Unexpected nvidia-smi output: {output!r}; no stages started')
        return rows

    inherited = selection is None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES') if inherited else selection
    selected = None
    if visible is not None:
        tokens = [token.strip() for token in visible.split(',')]
        if any(not token or token == '-1' for token in tokens):
            raise ValueError('No usable GPUs selected; specify --gpu with a physical index or GPU UUID')
        inventory = query('gpu', 'index,uuid,pci.bus_id')
        selected = []
        for token in tokens:
            if token.isdecimal():
                if inherited:
                    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
                        raise ValueError('Numeric CUDA_VISIBLE_DEVICES has ambiguous device ordering. '
                                         'Use --gpu <nvidia-smi physical index> or CUDA_VISIBLE_DEVICES=GPU-<UUID>.')
                    ordered = sorted(inventory, key=lambda row: tuple(
                        int(v, 16) for v in re.split(r'[:.]', row[2])))
                    matches = ordered[int(token):int(token) + 1]
                else:
                    matches = [row for row in inventory if int(row[0]) == int(token)]
            elif token.startswith('GPU-'):
                matches = [row for row in inventory if row[1].startswith(token)]
            else:
                raise ValueError(f'Unsupported GPU selector {token!r}; use a physical index or GPU UUID (MIG unsupported)')
            if len(matches) != 1:
                raise ValueError(f'Unknown or ambiguous GPU selector: {token!r}')
            uuid = matches[0][1]
            if uuid in selected:
                raise ValueError(f'Duplicate GPU selection: {token!r}')
            selected.append(uuid)
    processes = query('compute-apps', 'gpu_uuid,pid')
    busy = [dict(gpu_uuid=uuid, pid=pid) for uuid, pid in processes
            if selected is None or uuid in selected]
    # if busy:
    #     raise RuntimeError(f'GPU compute processes are still running: {busy}; no stages started')
    if selected is None:
        return {}
    print(f'GPU selection (CUDA logical order): {selected}', flush=True)
    return {'CUDA_VISIBLE_DEVICES': ','.join(selected)}


def make_plan(args, source):
    if args.stages != [s for s in STAGES if s in args.stages]:
        raise ValueError('Stages must be unique and follow pipeline order')
    target = Path(source["path"]).stem
    width, height = DIMENSIONS[args.resolution]
    layout = window_layout(source["frames"])
    if ('inference' in args.stages and any(p != 'baseline' for p in args.phases)
            and not layout['flowlong_supported']):
        raise ValueError("FlowLong requires at least two 49-frame windows; use single-window Vista4D for shorter timelines")
    return dict(target=target, resolution=args.resolution, source=source, width=width, height=height,
                baseline=window_layout(source["frames"],44), flowlong={'k1': layout},
                stages=args.stages, phases=args.phases,
                seed=args.seed, seg_keywords=args.seg_keywords, prompt=args.prompt,
                vista4d_folder=str(Path(args.vista4d_folder).expanduser().resolve()) if args.vista4d_folder else None,
                inference_root=str(ROOT/f"results/flowlong_eval/{target}_{args.resolution}_seed={args.seed}"),
                mode="execute" if args.execute else "plan_only",
                gpu=getattr(args, 'gpu', None),
                note="Full video; frame-index-uniform timing; audio is not propagated. No tmux or scheduling is implicit.")


def commands(plan):
    """Return commands without executing them or creating any paths."""
    target, res = plan["target"], plan["resolution"]
    baseline = split_directory(res, True)
    flowlong = split_directory(res)
    env = dict(RESOLUTION=res, SEED=str(plan["seed"]), SEEDS=str(plan["seed"]),
               HEIGHT=str(plan['height']), WIDTH=str(plan['width']),
               CLIP_FRAMES="49", TEMPORAL_ALIGNMENT="4", INCLUDE_TAIL="true",
               NUM_FRAMES="49", NUM_INFERENCE_STEPS="50", CFG_SCALE="5", SIGMA_SHIFT="5",
               PROMPT=plan['prompt'], FLOWLONG_STOCHASTIC_THRESHOLD="0.6",
               BASELINE_SPLITS_DIR=str(baseline), FLOWLONG_SPLITS_DIR=str(flowlong),
               FORCE="false", DRY_RUN="false")
    result=[]
    if plan.get('vista4d_folder'):
        env['VISTA4D_FOLDER'] = plan['vista4d_folder']
    def add(stage, command, **extra):
        result.append(dict(stage=stage, command=command, env={**env, **extra}))
    for stage in plan["stages"]:
        if stage=="split":
            for folder,overlap in ((baseline,5),(flowlong,25)):
                add(stage,["bash","scripts/test_video/split_video.sh",plan["source"]["path"]],
                    OUTPUT_DIR=str(folder),OVERLAP=str(overlap),START_FRAME="0",MAX_FRAMES=str(plan["source"]["frames"]))
        elif stage=="recon":
            # At execution time expand the freshly generated JSON manifest, not CSV.
            add(stage,["<each baseline JSON clip>","bash","scripts/test_video/recon_and_seg.sh"],
                RECON_METHOD="da3",DA3_PROCESS_RES=str(plan["width"]),SAVE_VIS="false",
                SEG_KEYWORDS=plan['seg_keywords'])
        elif stage=="stitch":
            add(stage,["bash","scripts/test_video/stitch_splits_smooth_and_slice.sh",target],
                SPLITS_DIR=str(baseline),INPUT_RESULT_ROOT="./results/single",
                OUTPUT_RESULT_ROOT="./results/stitched_single",FULL_RESULT_BASE="./results/full",
                TRANSLATION_SIGMA="8",ROTATION_SIGMA="10",FORCE_STITCH="false",OVERWRITE_SPLITS="false")
        elif stage=="render":
            add(stage,["bash","scripts/test_video/prepare_flowlong_ab_conditions.sh",target],
                STATIC_FRAME_STRIDE="4",RENDER_CHUNK_SIZE="4",
                OVERWRITE_FULL_RENDER="false",OVERWRITE_BASELINE_SPLITS="false",OVERWRITE_FLOWLONG_SPLITS="false")
        elif stage=="inference":
            add(stage,["bash","scripts/test_video/run_flowlong_stage4.sh",target],PHASES=" ".join(plan["phases"]))
    return result


def stage_environment():
    """Do not inherit a previous video's shell workflow overrides.

    Keep runtime environment (CUDA, conda, cache, credentials, proxy) unchanged.
    Resolve shell-script configuration variables through this CLI/defaults only.
    """
    names = set()
    for script in (ROOT/'scripts/test_video').glob('*.sh'):
        names.update(re.findall(r'^\s*([A-Z][A-Z0-9_]*)=', script.read_text(), re.M))
    names.difference_update({'PATH','HOME','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','PYTHONPATH',
                             'CONDA_PREFIX','CONDA_DEFAULT_ENV'})
    env = {k:v for k,v in os.environ.items() if k not in names}
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    return env


def validate_reconstruction(folder, frames, width, height):
    # Imported only during execution: plan-only mode never imports torch/CUDA.
    import numpy as np
    info = probe_source(folder/'video.mp4')
    if (info['frames'],info['width'],info['height']) != (frames,width,height):
        raise ValueError(f'Reconstruction video geometry mismatch: {folder}')
    for name in ('depths','dynamic_mask','sky_mask'):
        if len(list((folder/name).glob('*'))) != frames:
            raise ValueError(f'Incomplete {name}: {folder}')
    with np.load(folder/'cameras.npz') as cameras:
        if cameras['cam_c2w'].shape != (frames,4,4) or cameras['intrinsics'].shape[0] != frames:
            raise ValueError(f'Camera timeline mismatch: {folder}')
        if not all(np.isfinite(cameras[name]).all() for name in ('cam_c2w','intrinsics')):
            raise ValueError(f'Nonfinite cameras: {folder}')


def run(args):
    video=Path(args.video).expanduser().resolve()
    source=probe_source(video)
    plan=make_plan(args,source)
    jobs=commands(plan)
    print(json.dumps({**plan,"commands":jobs},indent=2,ensure_ascii=False),flush=True)
    if not args.execute:
        return plan
    # Check exactly the same devices that child processes will see.
    gpu_env = gpu_environment(getattr(args, 'gpu', None))
    # No implicit overwrite: protect old inputs even if another video has the same stem.
    for baseline in (False,True):
        p=split_directory(args.resolution,baseline)/f"{plan['target']}_splits_manifest.json"
        if p.exists():
            m=json.loads(p.read_text())
            if Path(m['input_path']).resolve()!=video or m['start_frame']!=0 or m['end_exclusive']!=source['frames']:
                raise ValueError(f"Existing manifest belongs to a different source/range: {p}")
            if 'split' in args.stages:
                raise FileExistsError(f"Existing splits found: {p}; omit the split stage to reuse them")
    for job in jobs:
        print(f"STAGE_START {job['stage']}", flush=True)
        env={**stage_environment(),**job['env'],**gpu_env}
        if job['stage']=='recon':
            m=json.loads((split_directory(args.resolution,True)/f"{plan['target']}_splits_manifest.json").read_text())
            for clip in m['clips']:
                folder=ROOT/'results/single'/Path(clip['output_path']).stem/'recon_and_seg'
                if folder.exists():
                    raise FileExistsError(f'Reconstruction already exists: {folder}; validate it and omit recon to reuse')
                subprocess.run(["bash","scripts/test_video/recon_and_seg.sh"],cwd=ROOT,
                    env={**env,"SOURCE_VIDEO":clip['output_path']},check=True)
                validate_reconstruction(folder,49,plan['width'],plan['height'])
                print(f"RECON_VALIDATED {folder}", flush=True)
        else:
            subprocess.run(job['command'],cwd=ROOT,env=env,check=True)
            if job['stage']=='stitch':
                folder=ROOT/f"results/full/{plan['target']}_stitched_{args.resolution}/recon_and_seg"
                validate_reconstruction(folder,source['frames'],plan['width'],plan['height'])
        print(f"STAGE_COMPLETE {job['stage']}", flush=True)
    return plan


def parser():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--config', help='YAML workflow; use --config FILE --help for its options')
    p.add_argument("--video",required=True)
    p.add_argument("--resolution",choices=DIMENSIONS,default="384p")
    p.add_argument("--seed",type=int,default=10027)
    p.add_argument("--vista4d-folder", help="Checkpoint directory with config.yaml and dit.pth, a single safetensors, or shard index")
    p.add_argument("--seg-keywords",default="person man woman hand phone bag backpack car stroller")
    p.add_argument("--prompt",default="A realistic handheld smartphone video of people in an everyday scene, with natural body motion, realistic lighting, stable camera motion, and detailed surroundings.")
    p.add_argument("--stages",nargs="+",choices=STAGES,default=list(STAGES[:5]))
    p.add_argument("--phases",nargs="+",choices=PHASES,default=list(PHASES))
    p.add_argument("--execute",action="store_true")
    p.add_argument("--gpu", help="Comma-separated nvidia-smi physical GPU indices or UUIDs; overrides CUDA_VISIBLE_DEVICES. Does not enable multi-GPU inference.")
    return p


if __name__=='__main__':
    if any(arg == '--config' or arg.startswith('--config=') for arg in sys.argv[1:]):
        from scripts.test_video.configured_video_experiment import parser as config_parser, run as config_run
        config_run(config_parser().parse_args())
    else:
        run(parser().parse_args())
