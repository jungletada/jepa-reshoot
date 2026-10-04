from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import torch

from jepa.adapter import AdapterConfig, JEPAAdapter
from jepa.features import FeatureCache, resample_time
from jepa.inference import load_jepa_inputs, validate_jepa_args
from jepa.predictor import PredictorConfig, TargetFeaturePredictor, feature_regression_loss, prepare_geometry
from jepa.teacher import FrozenVideoTeacher
from scripts.jepa.predict_features import predict


def sample_cache(role="source"):
    return FeatureCache(torch.randn(1, 8, 3, 2, 2), torch.tensor([.5, 2.5, 4.]), 5, (8, 8), "teacher-test", role)


def sample_geometry():
    cameras = torch.eye(4).repeat(1, 5, 1, 1)
    return {"source_depth": torch.ones(1, 5, 8, 8) * 2,
            "source_c2w": cameras.clone(), "target_c2w": cameras.clone(),
            "source_intrinsics": torch.tensor([8., 8., 4., 4.]).repeat(1, 5, 1),
            "target_intrinsics": torch.tensor([8., 8., 4., 4.]).repeat(1, 5, 1),
            "target_visibility": torch.ones(1, 5, 8, 8)}


class JEPAFeaturesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_frame_coordinate_interpolation_and_batch_expansion(self):
        features = torch.tensor([.5, 2.5, 4.]).reshape(1, 1, 3, 1, 1)
        result = resample_time(features, torch.tensor([.5, 2.5, 4.]), torch.tensor([0., 1., 2., 4.]))
        torch.testing.assert_close(result.flatten(), torch.tensor([.5, 1., 2., 4.]))
        cache = sample_cache()
        result = cache.for_generation(mode="source", num_frames=5, image_size=(8, 8), batch_size=2, encoder_id=cache.encoder_id)
        self.assertEqual(result.shape, (2, 8, 2, 2, 2))
        torch.testing.assert_close(result[0], result[1])

    def test_cache_provenance_rejects_oracle_in_predicted_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.pt"
            sample_cache("oracle").save(path)
            cache = FeatureCache.load(path)
            kwargs = dict(num_frames=5, image_size=(8, 8), batch_size=1, encoder_id=cache.encoder_id)
            with self.assertRaisesRegex(ValueError, "cache role"):
                cache.for_generation(mode="predicted", **kwargs)
            with self.assertRaisesRegex(ValueError, "encoder"):
                cache.for_generation(mode="oracle", **{**kwargs, "encoder_id": "wrong"})
            with self.assertRaisesRegex(ValueError, "length and resolution"):
                cache.for_generation(mode="oracle", **{**kwargs, "num_frames": 9})

    def test_teacher_freezes_and_pads_odd_clip_without_losing_last_frame(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4))
            def forward(self, x):
                self.last_input = x.detach().clone()
                return self.conv(x).flatten(2).transpose(1, 2)
        teacher = FrozenVideoTeacher(Encoder(), "test", (8, 8), 4, 2).train()
        cache = teacher.encode(torch.rand(1, 3, 5, 8, 8), "oracle")
        self.assertEqual(cache.features.shape, (1, 8, 3, 2, 2))
        torch.testing.assert_close(cache.frame_positions, torch.tensor([.5, 2.5, 4.]))
        torch.testing.assert_close(teacher.encoder.last_input[:, :, -1], teacher.encoder.last_input[:, :, -2])
        self.assertFalse(cache.features.requires_grad)
        self.assertFalse(teacher.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.encoder.parameters()))

    def test_geometry_backprojection_and_camera_query(self):
        cache, geometry = sample_cache(), sample_geometry()
        a = prepare_geometry(cache, geometry)
        torch.testing.assert_close(a["source_xyz"][0, :, 0, 0, 0], torch.tensor([-.25, -.25, 1.]))
        geometry["target_c2w"][:, :, 0, 3] = 2.
        b = prepare_geometry(cache, geometry)
        torch.testing.assert_close(a["source_xyz"], b["source_xyz"])
        torch.testing.assert_close(b["target_rays"][:, 0], torch.ones(1, 3, 2, 2))
        geometry["target_depth"] = geometry["source_depth"]
        with self.assertRaisesRegex(ValueError, "no target ground truth"):
            prepare_geometry(cache, geometry)

    def test_predictor_learns_teacher_loss_and_prediction_is_source_only(self):
        cache, geometry = sample_cache(), sample_geometry()
        model = TargetFeaturePredictor(PredictorConfig(8, 16, 2, 4, 1))
        inputs = prepare_geometry(cache, geometry)
        target = torch.randn_like(cache.features, requires_grad=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        initial = feature_regression_loss(model(cache.features, **inputs), target).item()
        for _ in range(12):
            optimizer.zero_grad()
            loss = feature_regression_loss(model(cache.features, **inputs), target)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial)
        self.assertIsNone(target.grad)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cache.save(folder / "source.pt")
            torch.save(geometry, folder / "geometry.pt")
            model.save(folder / "predictor.pt", cache.encoder_id)
            result = predict(folder / "predictor.pt", folder / "source.pt", folder / "geometry.pt", "cpu")
            self.assertEqual(result.role, "predicted")
            torch.testing.assert_close(result.features, model.eval()(cache.features, **inputs))
            sample_cache("oracle").save(folder / "source.pt")
            with self.assertRaisesRegex(ValueError, "SOURCE"):
                predict(folder / "predictor.pt", folder / "source.pt", folder / "geometry.pt", "cpu")

    def test_empty_geometry_and_empty_loss_weights_are_rejected(self):
        geometry = sample_geometry()
        geometry["source_depth"].zero_()
        with self.assertRaisesRegex(ValueError, "valid source depth"):
            prepare_geometry(sample_cache(), geometry)
        with self.assertRaisesRegex(ValueError, "positive total"):
            feature_regression_loss(torch.ones(1, 8, 2, 2, 2), torch.ones(1, 8, 2, 2, 2), torch.zeros(1, 2, 2, 2))

    def test_cli_is_opt_in_and_preserves_window_order(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_jepa_args(Namespace(jepa_mode="none", jepa_features=["oracle.pt"]))
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for i in range(3):
                cache = sample_cache("predicted")
                cache.features.fill_(i)
                path = Path(folder) / f"{i}.pt"
                cache.save(path)
                paths.append(str(path))
            args = Namespace(jepa_mode="predicted", jepa_features=paths, jepa_adapter="trained.pt",
                             jepa_scale=1., num_frames=5, height=8, width=8)
            inputs = load_jepa_inputs(args, Namespace(jepa_encoder_id=cache.encoder_id), 3, windowed=True)
            torch.testing.assert_close(inputs["jepa_features"][:, 0, 0, 0, 0], torch.arange(3).float())


if __name__ == "__main__":
    unittest.main()
