"""CPU-only configuration, command mapping, and resume contracts."""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from utils.video_config import DEFAULTS, load_config, validate, threshold_name, ROOT
from scripts.test_video.configured_video_experiment import (
    parser, configuration, make_plan, jobs, snapshot, run, FULL_STAGES,
    configured_environment,
)
from utils.video_experiment import probe_source

SOURCE = dict(path=str(ROOT/'data/1776148878076.mp4'), frames=462, fps=30.54,
              width=1920, height=1080)


class ConfigTest(unittest.TestCase):
    def test_three_video_presets_and_smoke(self):
        for target in ('1776148878076','1778135019043','1778134948379'):
            c=load_config(ROOT/'configs'/f'{target}.yaml')
            self.assertEqual(Path(c['video']['path']).stem,target)
            self.assertEqual(c['inference']['steps'],50)
            self.assertEqual(c['flowlong']['stochastic_thresholds'],[0.5,0.6])
        # Test short-run geometry without depending on a disposable smoke YAML.
        c=self.config()
        c['inference']['steps']=1
        c['flowlong'].update(matching_only=False,stochastic_thresholds=[0.6])
        p=make_plan(c,dict(SOURCE,path=c['video']['path'],frames=73))
        self.assertEqual(p['layouts']['k1']['windows'],2)
        self.assertEqual(p['phases'],['baseline','t0.6'])
        self.assertEqual(c['inference']['steps'],1)

    def config(self):
        c = deepcopy(DEFAULTS)
        c['video']['path'] = SOURCE['path']
        return validate(c)

    def test_load_real_and_cli_precedence(self):
        args = parser().parse_args(['--config','configs/1776148878076.yaml'])
        c = configuration(args)
        self.assertEqual(c['inference']['steps'],50)
        args = parser().parse_args(['--config','configs/1776148878076.yaml',
                                   '--resolution','720p','--seed','42','--steps','2'])
        c = configuration(args)
        self.assertEqual(c['video']['resolution'],'720p')
        self.assertEqual(c['inference']['seed'],42)
        self.assertEqual(c['inference']['steps'],2)
        with patch.dict(os.environ, {'RESOLUTION':'720p','NUM_INFERENCE_STEPS':'1','OUTPUT_FOLDER':'bad'}):
            c = configuration(parser().parse_args(['--config','configs/1776148878076.yaml']))
        self.assertEqual(c['video']['resolution'],'384p')
        self.assertEqual(c['inference']['steps'],50)

    def test_yaml_rejects_duplicates_unknown_keys_and_bad_types(self):
        invalid = ['video: {path: a.mp4}\nvideo: {path: b.mp4}',
                   'video: {path: a.mp4, resoluton: 720p}',
                   'video: {path: a.mp4}\ninference: {tile_vae: "false"}',
                   'video: {path: a.mp4}\ninference: {steps: true}',
                   'video: {path: a.mp4}\nflowlong: {stochastic_thresholds: [.nan]}',
                   'video: {path: a.mp4}\nflowlong: {stochastic_thresholds: [0]}',
                   'video: {path: a.mp4}\nsubsample: {stochastic_threshold: 0}',
                   'video: {path: a.mp4}\noutputs: {run_name: ../bad}',
                   'video: {path: a.mp4}\nwindows: {flowlong_overlap: 24}',
                   'video: {path: a.mp4}\nsubsample: {factors: [2, 2]}',
                   'video: {path: a.mp4}\ninterpolation: {tail_policy: hold}',
                   'video: {path: a.mp4}\nevaluation: {reference_variant: matching_only}',
                   '!!python/object/apply:os.system ["false"]', '']
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.yaml'
            for text in invalid:
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises((ValueError,yaml.YAMLError)):
                        load_config(path)

    def test_relative_paths_are_repo_relative(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.yaml'
            path.write_text('video: {path: data/my_video.mp4}\nmodels: {vista4d: checkpoints/custom}')
            c = load_config(path)
        self.assertEqual(c['video']['path'],str(ROOT/'data/my_video.mp4'))
        self.assertEqual(c['models']['vista4d'],str(ROOT/'checkpoints/custom'))

    def test_no_hidden_shell_overrides(self):
        with patch.dict(os.environ,dict(RESULT_ROOT='/wrong',NEGATIVE_PROMPT='wrong',
                        EXTRA_ARGS='--wrong',OUTPUT_FOLDER='/wrong',HF_HOME='/cache',CUDA_VISIBLE_DEVICES='GPU-abc')):
            env=configured_environment()
        for key in ('RESULT_ROOT','NEGATIVE_PROMPT','EXTRA_ARGS','OUTPUT_FOLDER'):
            self.assertNotIn(key,env)
        self.assertEqual(env['HF_HOME'],'/cache')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-abc')

    def test_all_stage_paths_and_parameters(self):
        for res in ('384p','720p'):
            c = self.config()
            c['video']['resolution']=res
            c['inference'].update(steps=7,cfg_scale=3.5,sigma_shift=2.0,microbatch_size=2)
            c['preprocess'].update(da3_process_res=512,segmentation_keywords=['person','car'])
            c['camera'].update(translation_sigma=3,rotation_sigma=6)
            c['render'].update(static_frame_stride=8,chunk_size=2)
            c['flowlong']['stochastic_thresholds']=[0.4,0.65]
            p=make_plan(c,SOURCE,FULL_STAGES)
            commands=jobs(p)
            env=commands[0]['env']
            self.assertEqual(env['NUM_INFERENCE_STEPS'],'7')
            self.assertEqual(env['CFG_SCALE'],'3.5')
            self.assertEqual(env['SIGMA_SHIFT'],'2.0')
            self.assertEqual(env['DA3_PROCESS_RES'],'512')
            self.assertEqual(env['SEG_KEYWORDS'],'person car')
            self.assertEqual(env['TRANSLATION_SIGMA'],'3')
            self.assertEqual(env['STATIC_FRAME_STRIDE'],'8')
            self.assertIn('720p49' if res=='720p' else '384p49',env['VISTA4D_FOLDER'])
            for job in commands:
                for key in ('BASELINE_SPLITS_DIR','FLOWLONG_SPLITS_DIR','FULL_RESULT_BASE',
                            'FULL_SEQUENCE_ROOT','FULL_RECON_FOLDER','BASELINE_CONDITION_ROOT',
                            'FLOWLONG_RESULT_ROOT','EVAL_ROOT','LOG_ROOT'):
                    self.assertTrue(Path(job['env'][key]).is_relative_to(p['paths']['root']),key)
            infer=next(j for j in commands if j['stage']=='inference')
            self.assertEqual(infer['env']['PHASES'],'baseline matching_only t0.4 t0.65')

    def test_single_clip_derives_name_and_paths(self):
        c=self.config()
        c['video']['mode']='single_clip'
        c['render']['mode']='auto'
        c['single_clip']['start_frame']=17
        c=validate(c)
        p=make_plan(c,SOURCE)
        self.assertEqual(p['clip_name'],'1776148878076_frames000017_000065_384p49')
        commands=jobs(p)
        self.assertEqual([j['stage'] for j in commands],['split','recon','smooth','render','inference'])
        self.assertEqual(commands[1]['env']['EXAMPLE'],p['clip_name'])
        self.assertEqual(commands[1]['env']['CLIP_NAME'],p['clip_name'])
        self.assertIn('--start_frame',commands[0]['command'])
        c['single_clip']['name']='custom_clip'
        self.assertEqual(make_plan(c,SOURCE)['clip_name'],'custom_clip')
        with self.assertRaises(ValueError):
            make_plan(c,dict(SOURCE,frames=60))

    def test_stage_validation(self):
        for stages in (['render','split'],['smooth'],['split','split'],
                       ['evaluate'],['subsample_inference'],['interpolate'],['postprocess']):
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                make_plan(self.config(),SOURCE,stages)
        with self.assertRaisesRegex(ValueError,'two'):
            make_plan(self.config(),dict(SOURCE,frames=49),['inference'])
        c=self.config()
        c['flowlong'].update(matching_only=False,stochastic_thresholds=[])
        self.assertEqual(make_plan(c,dict(SOURCE,frames=49),['inference'])['phases'],['baseline'])

    def test_removed_cli_options_are_rejected(self):
        for option in (['--stages','evaluate'], ['--vfi-python','python'], ['--factors','2']):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser().parse_args(['--config','configs/example.yaml',*option])

    def test_plan_is_nonmutating_and_gpu_free(self):
        args=parser().parse_args(['--config','configs/1776148878076.yaml'])
        with patch('scripts.test_video.configured_video_experiment.probe_source',return_value=SOURCE), \
             patch('subprocess.run') as execute, patch('subprocess.check_output') as gpu, \
             patch.object(Path,'mkdir') as mkdir, contextlib.redirect_stdout(io.StringIO()):
            run(args)
            execute.assert_not_called(); gpu.assert_not_called(); mkdir.assert_not_called()

    def test_snapshot_blocks_config_source_and_untracked_outputs(self):
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'video.mp4'; source.write_bytes(b'original')
            p=make_plan(self.config(),dict(SOURCE,path=str(source)),['split'])
            p['paths']['root']=str(Path(tmp)/'run')
            snapshot(p); snapshot(p)
            p['config']['inference']['seed']+=1
            with self.assertRaisesRegex(ValueError,'changed'):
                snapshot(p)
            p['config']['inference']['seed']-=1
            source.write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError,'changed'):
                snapshot(p)
            p['paths']['root']=str(Path(tmp)/'legacy')
            Path(p['paths']['root']).mkdir()
            (Path(p['paths']['root'])/'result.mp4').touch()
            with self.assertRaisesRegex(ValueError,'Untracked'):
                snapshot(p)

    def test_stage4_custom_parameters_dry_run(self):
        env={**os.environ,'DRY_RUN':'true','RESOLUTION':'384p','PHASES':'baseline t0.65',
             'ALLOW_CUSTOM_INFERENCE':'true','NUM_INFERENCE_STEPS':'7','CFG_SCALE':'3.5','SIGMA_SHIFT':'2'}
        result=subprocess.run(['bash','scripts/test_video/run_flowlong_stage4.sh','config_test'],
                              cwd=ROOT,env=env,text=True,capture_output=True,check=True)
        self.assertIn('--num_inference_steps 7',result.stdout)
        self.assertIn('flowlong_t0p65',result.stdout)
        self.assertNotIn('==== Stage 4: flowlong_t0p5 ====',result.stdout)
        self.assertEqual(threshold_name(.65),'flowlong_t0p65')

    def test_stage4_rejects_removed_phase_before_creating_outputs(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            env={**os.environ,'PHASES':'evaluate','DRY_RUN':'false',
                 'EVAL_ROOT':str(output),'NUM_INFERENCE_STEPS':'50',
                 'CFG_SCALE':'5','SIGMA_SHIFT':'5','RESOLUTION':'384p'}
            result=subprocess.run(['bash','scripts/test_video/run_flowlong_stage4.sh','config_test'],
                                  cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Unknown phase: evaluate',result.stderr)
            self.assertFalse(output.exists())

    def test_real_cpu_split_execute_and_verified_resume(self):
        # All outputs stay under /tmp; no GPU/model/real experiment involved.
        with TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            video=tmp/'synthetic.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x36:rate=25',
                            '-frames:v','77','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
            config_path=tmp/'config.yaml'
            config_path.write_text(yaml.safe_dump({'video':{'path':str(video)}}))
            args=parser().parse_args(['--config',str(config_path),'--stages','split','--execute'])
            plan=make_plan(configuration(args),probe_source(video),['split'])
            old_root=plan['paths']['root']
            plan['paths']={k:str(tmp/'run')+v[len(old_root):] for k,v in plan['paths'].items()}
            actual_run=subprocess.run
            stage_calls=[]
            def quiet_run(*pos,**kw):
                if pos[0][0]=='bash':
                    stage_calls.append(pos[0])
                    kw.update(capture_output=True,text=True)
                return actual_run(*pos,**kw)
            with patch('scripts.test_video.configured_video_experiment.make_plan',return_value=plan), \
                 patch('scripts.test_video.run_video_experiment.gpu_environment') as gpu, \
                 patch('subprocess.run',side_effect=quiet_run) as execute, \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args)
                self.assertEqual(len(stage_calls),2)
                gpu.assert_not_called()
                run(args)
                self.assertEqual(len(stage_calls),2)
                for kind, stride in (('baseline',44),('flowlong',24)):
                    path=Path(plan['paths'][kind+'_splits'])/'synthetic_splits_manifest.json'
                    manifest=json.loads(path.read_text())
                    self.assertEqual(manifest['stride'],stride)
                    self.assertEqual(manifest['end_exclusive'],77)
                    self.assertTrue(all(clip['start_frame'] % 4 == 0 for clip in manifest['clips']))
                    for clip in manifest['clips']:
                        self.assertEqual(probe_source(Path(clip['output_path']))['frames'],49)
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'Changed/missing'):
                    run(args)


if __name__ == '__main__':
    unittest.main()
