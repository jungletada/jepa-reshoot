"""Wait for a validated predecessor, then run a frozen workflow command once.

No GPU work occurs during waiting. A config file is an explicit launch request;
this module has no implicit default video, predecessor or inference phases.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from utils.video_experiment import ROOT, probe_source


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19]  # Linux stat field 22, process start ticks
    except FileNotFoundError:
        return None


def predecessor_state(config):
    status = Path(config['status']).read_text() if Path(config['status']).exists() else ''
    if config['start_marker'] not in status:
        raise RuntimeError('Predecessor status identity changed or missing')
    last = status.strip().splitlines()[-1]
    if 'FAILED' in last:
        raise RuntimeError(f'Predecessor failed: {last}')
    if last.endswith(' ALL_COMPLETE'):
        return 'complete'
    if process_identity(config['pid']) != config['pid_start_ticks']:
        raise RuntimeError('Predecessor process exited without successful validation')
    return 'running'


def validate_predecessor(config):
    report = json.loads(Path(config['report']).read_text())
    for key, value in config['expected'].items():
        if report.get(key) != value:
            raise ValueError(f'Predecessor report mismatch: {key}')
    exp = report['experiment']
    if exp['num_inference_steps'] != 50 or exp['stochastic_enabled'] is not False:
        raise ValueError('Expected 50-step matching-only predecessor')
    rows = report['pipeline']['steps']
    if len(rows) != 50 or any(r['overlap_after_max_abs'] != 0 or r['stochastic'] for r in rows):
        raise ValueError('Predecessor overlap matching validation failed')
    if sha256(config['video']) != report['output_video_sha256']:
        raise ValueError('Predecessor output hash mismatch')
    info = probe_source(config['video'])
    if (info['frames'],info['width'],info['height']) != (report['output_frames'],report['width'],report['height']):
        raise ValueError('Predecessor decoded geometry mismatch')


def sleep_until(deadline):
    while time.time() < deadline:
        time.sleep(min(60, max(0, deadline-time.time())))


def validate_successor(config):
    expected = config['successor']
    for phase in expected['phases']:
        folder_name = {'t0.5':'flowlong_t0p5','t0.6':'flowlong_t0p6'}.get(phase,phase)
        folder = Path(expected['output_root'])/folder_name
        seed = expected['seed']
        baseline = phase == 'baseline'
        report_path = folder/f"{'baseline' if baseline else 'flowlong'}_report_seed={seed}.json"
        video = folder/f"video_seed={seed}{'_center_cut' if baseline else ''}.mp4"
        report = json.loads(report_path.read_text())
        for key in ('output_frames','resolution','width','height'):
            if report[key] != expected[key]:
                raise ValueError(f'{phase} output mismatch: {key}')
        for key,value in dict(seed=seed,num_inference_steps=50,cfg_scale=5,sigma_shift=5).items():
            if report['experiment'][key] != value:
                raise ValueError(f'{phase} parameter mismatch: {key}')
        if not baseline:
            rows = report['pipeline']['steps']
            if len(rows) != 50 or any(r['overlap_after_max_abs'] != 0 for r in rows):
                raise ValueError(f'{phase} matching failed')
            if report['experiment']['stochastic_enabled'] != (phase != 'matching_only'):
                raise ValueError(f'{phase} stochastic setting mismatch')
            if phase.startswith('t') and report['experiment']['stochastic_threshold'] != float(phase[1:]):
                raise ValueError(f'{phase} threshold mismatch')
        if sha256(video) != report['output_video_sha256']:
            raise ValueError(f'{phase} output hash mismatch')
        info = probe_source(video)
        if (info['frames'],info['width'],info['height']) != (expected['output_frames'],expected['width'],expected['height']):
            raise ValueError(f'{phase} decoded geometry mismatch')


def run(config_path):
    config = json.loads(config_path.read_text())
    folder = config_path.parent
    with (folder/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def mark(state, **details):
            event = {'time':datetime.now().astimezone().isoformat(), 'state':state, **details}
            with (folder/'queue_status.jsonl').open('a') as handle:
                handle.write(json.dumps(event,ensure_ascii=False)+'\n')
            print(json.dumps(event,ensure_ascii=False),flush=True)
        if (folder/'started.json').exists():
            raise RuntimeError('Launch already attempted; inspect outputs before creating a new queue')
        try:
            deadline = datetime.fromisoformat(config['not_before']).timestamp()
            mark('WAITING_INITIAL_DELAY', first_check=config['not_before'], poll_seconds=config['poll_seconds'])
            sleep_until(deadline)
            validated = False
            while True:
                state = predecessor_state(config['predecessor'])
                if state == 'complete':
                    if not validated:
                        validate_predecessor(config['predecessor'])
                        validated = True
                        mark('PREDECESSOR_VALIDATED')
                    busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                                    '--format=csv,noheader'],text=True).strip()
                    if not busy:
                        break
                    mark('WAITING_GPU_IDLE', compute_pids=busy)
                else:
                    mark('WAITING_PREDECESSOR')
                sleep_until(time.time()+config['poll_seconds'])
            if sha256(config['source_video']) != config['source_sha256']:
                raise ValueError('Successor source video changed while queued')
            # Exclusive marker prevents accidental double launch after a restart.
            with (folder/'started.json').open('x') as handle:
                json.dump({'command':config['command'],'time':datetime.now().astimezone().isoformat()},handle)
            mark('SUCCESSOR_START', command=config['command'])
            with (folder/'pipeline.log').open('a') as log:
                subprocess.run(config['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            mark('SUCCESSOR_OUTPUT_VALIDATION')
            validate_successor(config)
            mark('SUCCESSOR_COMPLETE')
        except BaseException as exc:
            mark('FAILED', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config.resolve())
