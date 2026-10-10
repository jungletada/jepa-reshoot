"""Unified entry point tests: no GPU jobs or source-video outputs."""
import contextlib
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.test_video.run_video_experiment import parser, make_plan, commands, run, stage_environment, gpu_environment, STAGES

SOURCE = dict(path='/tmp/1776148878076.mp4', frames=462, fps=30.5378270992,
              width=1920,height=1080)


class VideoExperimentTest(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(['--video',SOURCE['path'],*extra])

    def test_profiles_and_full_video_paths(self):
        for res,width,height in [('384p',672,384),('720p',1280,720)]:
            p=make_plan(self.args('--resolution',res,'--stages',*STAGES),SOURCE)
            self.assertEqual((p['width'],p['height']),(width,height))
            self.assertEqual(p['baseline']['windows'],11)
            self.assertEqual(p['flowlong']['k1']['windows'],19)
            jobs=commands(p)
            self.assertEqual(len(jobs),6)
            self.assertEqual(jobs[0]['env']['MAX_FRAMES'],'462')
            self.assertEqual(jobs[-1]['env']['RESOLUTION'],res)
            self.assertTrue(all(j['env']['FORCE']=='false' for j in jobs))

    def test_plan_only_has_no_stage_or_gpu_calls_or_mkdir(self):
        with patch('scripts.test_video.run_video_experiment.probe_source',return_value=SOURCE), \
             patch('subprocess.run') as stage, patch('subprocess.check_output') as gpu, \
             patch.object(Path,'mkdir') as mkdir, contextlib.redirect_stdout(io.StringIO()):
            result=run(self.args())
            self.assertEqual(result['mode'],'plan_only')
            stage.assert_not_called(); gpu.assert_not_called(); mkdir.assert_not_called()

    def test_checkpoint_folder_explicit_cli_override(self):
        plan = make_plan(self.args('--vista4d-folder', '/tmp/custom-sharded',
                                   '--stages', 'inference'), SOURCE)
        for job in commands(plan):
            self.assertEqual(job['env']['VISTA4D_FOLDER'], str(Path('/tmp/custom-sharded').resolve()))

    def test_execute_propagates_gpu_setup_failure_before_stages(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch('scripts.test_video.run_video_experiment.probe_source',return_value=SOURCE), \
             patch('scripts.test_video.run_video_experiment.gpu_environment',
                   side_effect=RuntimeError('GPU setup failed')), \
             patch('subprocess.run') as stage, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'GPU setup failed'):
                run(self.args('--execute'))
            stage.assert_not_called()

    def test_gpu_selection_ignores_other_cards(self):
        inventory = '0, GPU-aaa, 00000000:02:00.0\n1, GPU-bbb, 00000000:01:00.0\n'
        with patch('subprocess.check_output', side_effect=[inventory, 'GPU-aaa, 123\n']):
            self.assertEqual(gpu_environment('1'), {'CUDA_VISIBLE_DEVICES': 'GPU-bbb'})
        with patch('subprocess.check_output', side_effect=[inventory, 'GPU-bbb, 123\n']):
            # Busy refusal was explicitly disabled in the existing workflow.
            # Selection must still restrict children to the requested GPU.
            self.assertEqual(gpu_environment('1'), {'CUDA_VISIBLE_DEVICES': 'GPU-bbb'})

    def test_gpu_uuid_order_and_invalid_selection(self):
        inventory = '0, GPU-aaa, 00000000:02:00.0\n1, GPU-bbb, 00000000:01:00.0\n'
        with patch('subprocess.check_output', side_effect=[inventory, '']):
            self.assertEqual(gpu_environment('GPU-b,0')['CUDA_VISIBLE_DEVICES'], 'GPU-bbb,GPU-aaa')
        for selection in ('8', '0,GPU-aaa', 'GPU-', '-1', '', 'MIG-foo'):
            with self.subTest(selection=selection), patch('subprocess.check_output', return_value=inventory):
                with self.assertRaises(ValueError):
                    gpu_environment(selection)

    def test_inherited_gpu_environment(self):
        inventory = '0, GPU-aaa, 00000000:02:00.0\n1, GPU-bbb, 00000000:01:00.0\n'
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-bbb'}, clear=True), \
             patch('subprocess.check_output', side_effect=[inventory, 'GPU-aaa, 123\n']):
            self.assertEqual(gpu_environment()['CUDA_VISIBLE_DEVICES'], 'GPU-bbb')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}, clear=True), \
             patch('subprocess.check_output', side_effect=[inventory, '']):
            self.assertEqual(gpu_environment()['CUDA_VISIBLE_DEVICES'], 'GPU-bbb')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}, clear=True), \
             patch('subprocess.check_output', return_value=inventory):
            with self.assertRaisesRegex(ValueError, 'ambiguous device ordering'):
                gpu_environment()

    def test_selected_gpu_is_propagated_to_stages(self):
        with patch('scripts.test_video.run_video_experiment.probe_source', return_value=SOURCE), \
             patch('scripts.test_video.run_video_experiment.gpu_environment',
                   return_value={'CUDA_VISIBLE_DEVICES': 'GPU-bbb'}) as guard, \
             patch.object(Path, 'exists', return_value=False), \
             patch('subprocess.run') as stage, contextlib.redirect_stdout(io.StringIO()):
            run(self.args('--execute', '--gpu', '1', '--stages', 'split'))
            guard.assert_called_once_with('1')
            self.assertEqual(stage.call_count, 2)
            for call in stage.call_args_list:
                self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-bbb')

    def test_stage_order_and_small_video_rejected(self):
        with self.assertRaisesRegex(ValueError,'order'):
            make_plan(self.args('--stages','render','split'),SOURCE)
        with self.assertRaisesRegex(ValueError,'two'):
            make_plan(self.args('--stages','inference'),dict(SOURCE,frames=49))

    def test_environment_does_not_inherit_previous_video(self):
        with patch.dict(os.environ,{'RESULT_ROOT':'wrong','VISTA4D_FOLDER':'wrong',
                                    'OUTPUT_DIR':'wrong','SOURCE_VIDEO':'wrong','FORCE':'true',
                                    'CUDA_VISIBLE_DEVICES':'0','HF_HOME':'/tmp/hf'}):
            env=stage_environment()
            for key in ('RESULT_ROOT','VISTA4D_FOLDER','OUTPUT_DIR','SOURCE_VIDEO','FORCE'):
                self.assertFalse(key in env, f'Leaked workflow override: {key}')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
            self.assertEqual(env['HF_HOME'],'/tmp/hf')

    def test_removed_stage_and_phase_are_rejected(self):
        for option in (['--stages','subsample_prepare'], ['--stages','postprocess'],
                       ['--phases','evaluate'], ['--factors','2']):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*option)

    def test_short_baseline_does_not_require_flowlong(self):
        plan=make_plan(self.args('--stages','inference','--phases','baseline'),
                       dict(SOURCE,frames=49))
        self.assertEqual(plan['baseline']['windows'],1)
        self.assertEqual(commands(plan)[0]['env']['PHASES'],'baseline')


if __name__=='__main__': unittest.main()
