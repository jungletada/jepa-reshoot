import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.test_video.queue_video_experiment import predecessor_state, validate_predecessor, run
from scripts.test_video.run_video_experiment import parser, make_plan, commands


class QueueTests(unittest.TestCase):
    def test_waits_for_success_and_idle_then_launches_once(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json'
            p.write_text(json.dumps(dict(not_before=datetime.now(timezone.utc).isoformat(),
                poll_seconds=1800,predecessor={},source_video='source',source_sha256='hash',
                command=['not-a-real-command'],successor={})))
            module='scripts.test_video.queue_video_experiment.'
            with patch(module+'sleep_until') as wait, \
                 patch(module+'predecessor_state',side_effect=['running','complete','complete']), \
                 patch(module+'validate_predecessor') as validate, \
                 patch(module+'subprocess.check_output',side_effect=['999\n','']), \
                 patch(module+'sha256',return_value='hash'), \
                 patch(module+'subprocess.run') as launch, \
                 patch(module+'validate_successor'), contextlib.redirect_stdout(io.StringIO()):
                run(p)
                self.assertEqual(wait.call_count,3)
                validate.assert_called_once()
                launch.assert_called_once()
                with self.assertRaisesRegex(RuntimeError,'already attempted'): run(p)
            rows=[json.loads(x) for x in (p.parent/'queue_status.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['state'],'SUCCESSOR_COMPLETE')

    def test_four_phases_without_extra_variant_or_evaluation(self):
        args=parser().parse_args(['--video','new.mp4','--resolution','384p',
                                '--phases','baseline','matching_only','t0.5','t0.6'])
        plan=make_plan(args,dict(path='/tmp/new.mp4',frames=462,fps=30,width=1920,height=1080))
        self.assertEqual(plan['baseline']['windows'],11)
        self.assertEqual(plan['flowlong']['k1']['windows'],19)
        jobs=commands(plan)
        self.assertEqual(jobs[-1]['env']['PHASES'],'baseline matching_only t0.5 t0.6')

    def test_only_successful_marker_allows_readiness(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'status'
            cfg=dict(status=str(path),start_marker='old-run START',pid=123,pid_start_ticks='456')
            with patch('scripts.test_video.queue_video_experiment.process_identity',return_value='456'):
                path.write_text('old-run START\n')
                self.assertEqual(predecessor_state(cfg),'running')
                path.write_text('old-run START\n[now] ALL_COMPLETE\n')
                self.assertEqual(predecessor_state(cfg),'complete')
                path.write_text('old-run START\n[now] FAILED exit=1\n')
                with self.assertRaises(RuntimeError): predecessor_state(cfg)
            path.write_text('old-run START\n')
            with patch('scripts.test_video.queue_video_experiment.process_identity',return_value=None):
                with self.assertRaises(RuntimeError): predecessor_state(cfg)

    def test_predecessor_hash_and_matching_validation(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json'
            report=dict(output_frames=10,width=20,height=30,output_video_sha256='good',
                        experiment=dict(num_inference_steps=50,stochastic_enabled=False),
                        pipeline={'steps':[dict(overlap_after_max_abs=0,stochastic=False)]*50})
            path.write_text(json.dumps(report))
            cfg=dict(report=str(path),video='video',expected={'output_frames':10})
            with patch('scripts.test_video.queue_video_experiment.sha256',return_value='bad'):
                with self.assertRaisesRegex(ValueError,'hash'): validate_predecessor(cfg)
            with patch('scripts.test_video.queue_video_experiment.sha256',return_value='good'), \
                 patch('scripts.test_video.queue_video_experiment.probe_source',return_value=dict(frames=10,width=20,height=30)):
                validate_predecessor(cfg)


if __name__ == '__main__': unittest.main()
