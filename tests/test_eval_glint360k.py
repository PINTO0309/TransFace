from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import eval_glint360k


def _encode_jpg(color: int) -> bytes:
    image = np.full((112, 112, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode test image")
    return encoded.tobytes()


def _write_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, BytesIO(data))


def _build_test_shard(path: Path) -> None:
    members = [
        ("0_0.cls", b"0\n"),
        ("0_0.jpg", _encode_jpg(10)),
        ("0_1.cls", b"0\n"),
        ("0_1.jpg", _encode_jpg(20)),
        ("1_0.cls", b"1\n"),
        ("1_0.jpg", _encode_jpg(30)),
        ("1_1.cls", b"1\n"),
        ("1_1.jpg", _encode_jpg(40)),
        ("1_2.cls", b"1\n"),
        ("1_2.jpg", _encode_jpg(50)),
        ("2_0.cls", b"2\n"),
        ("2_0.jpg", _encode_jpg(60)),
    ]
    with tarfile.open(path, "w") as tar:
        for name, data in members:
            _write_member(tar, name, data)


def _fake_feature_extractor(
    refs: list[tuple[str, str, int, int]],
    data_root: Path,
    model_path: Path,
    network: str,
    batch_size: int,
    use_flip_test: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    del data_root, model_path, network, batch_size, use_flip_test
    features = np.zeros((len(refs), 4), dtype=np.float32)
    valid_mask = np.ones(len(refs), dtype=bool)
    for index, ref in enumerate(refs):
        member_name = ref[1]
        identity = int(Path(member_name).stem.split("_")[0])
        if identity == 0:
            feature = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            feature = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        features[index] = feature
    return features, valid_mask, 0


class EvalGlint360kTests(unittest.TestCase):
    def test_build_index_and_cache_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            shard_path = tmpdir_path / "glint360k_train-000.tar"
            cache_path = tmpdir_path / "index.pkl"
            _build_test_shard(shard_path)

            cold = eval_glint360k.load_or_build_index(tmpdir_path, cache_path)
            warm = eval_glint360k.load_or_build_index(tmpdir_path, cache_path)

        self.assertEqual(cold["eligible_id_list"], ["0", "1"])
        self.assertEqual(cold, warm)
        self.assertEqual(len(cold["identity_to_refs"]["0"]), 2)
        self.assertEqual(len(cold["identity_to_refs"]["1"]), 2)
        self.assertEqual(len(cold["identity_to_refs"]["2"]), 1)
        self.assertEqual(cold["identity_to_refs"]["1"][0][1], "1_0.jpg")
        self.assertEqual(cold["identity_to_refs"]["1"][1][1], "1_1.jpg")

    def test_negative_indices_are_deterministic(self) -> None:
        indices = eval_glint360k.build_negative_gallery_indices(5, 3)
        expected = np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 0],
                [4, 0, 1],
                [0, 1, 2],
            ],
            dtype=np.int64,
        )
        self.assertTrue(np.array_equal(indices, expected))

    def test_compute_roc_metrics(self) -> None:
        scores = np.array([0.95, 0.9, 0.1, 0.05], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=np.uint8)
        tar_by_far, fpr, tpr, auc_value = eval_glint360k.compute_roc_metrics(scores, labels, far_targets=(0.5,))

        self.assertAlmostEqual(auc_value, 1.0)
        self.assertAlmostEqual(tar_by_far["0.5"], 1.0)
        self.assertEqual(fpr[0], 0.0)
        self.assertEqual(tpr[0], 0.0)

    def test_compute_accuracy_at_far(self) -> None:
        scores = np.array([0.95, 0.9, 0.1, 0.05], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=np.uint8)
        accuracy_by_far, threshold_by_far = eval_glint360k.compute_accuracy_at_far(scores, labels, far_targets=(0.5,))

        self.assertAlmostEqual(accuracy_by_far["0.5"], 0.75)
        self.assertGreaterEqual(threshold_by_far["0.5"], 0.1 - 1e-6)
        self.assertLessEqual(threshold_by_far["0.5"], 0.1 + 1e-6)

    def test_evaluate_glint360k_writes_outputs_with_fake_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            data_root = tmpdir_path / "data"
            result_dir = tmpdir_path / "result"
            data_root.mkdir()
            result_dir.mkdir()
            _build_test_shard(data_root / "glint360k_train-000.tar")

            args = SimpleNamespace(
                model_prefix=str(tmpdir_path / "dummy.pt"),
                data_root=str(data_root),
                result_dir=str(result_dir),
                network="vit_s_dp005_mask_0",
                batch_size=2,
                job="unit",
                index_cache=None,
                num_negatives_per_id=1,
                seed=42,
                no_flip_test=False,
                chunk_size=2,
            )
            summary = eval_glint360k.evaluate_glint360k(args, feature_extractor=_fake_feature_extractor)

            job_dir = result_dir / "unit"
            summary_path = job_dir / "summary.json"
            scores_path = job_dir / "scores.npy"
            labels_path = job_dir / "labels.npy"
            roc_path = job_dir / "roc.pdf"

            self.assertTrue(summary_path.exists())
            self.assertTrue(scores_path.exists())
            self.assertTrue(labels_path.exists())
            self.assertTrue(roc_path.exists())

            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            saved_scores = np.load(scores_path)
            saved_labels = np.load(labels_path)

        self.assertEqual(summary["num_evaluated_identities"], 2)
        self.assertEqual(saved_summary["num_positive_pairs"], 2)
        self.assertEqual(saved_summary["num_negative_pairs"], 2)
        self.assertAlmostEqual(saved_summary["accuracy_at_far"]["0.1"], 1.0)
        self.assertIn("threshold_at_far", saved_summary)
        self.assertEqual(saved_scores.shape, (4,))
        self.assertEqual(saved_labels.tolist(), [1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
