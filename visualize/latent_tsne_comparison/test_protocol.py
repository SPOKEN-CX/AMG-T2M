"""Small protocol tests: split isolation, source independence and caption time bounds."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze import latent_metrics, paired_bootstrap
from common import FIELDS, read_rows, validate_rows, write_rows
from features import normalize
from prepare import captions_for, finalize


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "train.txt").write_text("000001\nM000001\n")
        (self.root / "val.txt").write_text("000002\n")
        (self.root / "test.txt").write_text("000003\nM000003\n000004\n000005\n000006\nM000001\n")
        self.cfg = {"dataset_root": str(self.root), "window_size": 4,
                    "architecture": {"input_dim": 263}, "classes": [{"name": "a"}, {"name": "b"}],
                    "samples_per_class": 30, "selection_seed": 42}
        self.rows = []
        for ident, label in [("000003", "a"), ("000004", "a"), ("000005", "b"), ("000006", "b")]:
            row = dict.fromkeys(FIELDS, "")
            row.update(sample_id=ident, motion_id=ident, source_id=ident, label=label,
                       start_frame="2", end_frame="6", annotation_start="0", annotation_end="8",
                       motion_frames="8", review_status="pending", caption_votes="2")
            self.rows.append(row)

    def tearDown(self):
        self.tmp.cleanup()

    def test_balanced_pending_is_only_allowed_for_draft(self):
        self.assertEqual(validate_rows(self.rows, self.cfg, check_files=False), {"a": 2, "b": 2})
        with self.assertRaisesRegex(ValueError, "not been accepted"):
            validate_rows(self.rows, self.cfg, require_reviewed=True, check_files=False)

    def test_mirror_source_cannot_be_counted_twice(self):
        rows = copy.deepcopy(self.rows)
        rows[1].update(motion_id="M000003", source_id="000003")
        with self.assertRaisesRegex(ValueError, "Duplicate or invalid canonical"):
            validate_rows(rows, self.cfg, check_files=False)

    def test_train_mirror_leakage_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0].update(motion_id="M000001", source_id="000001")
        with self.assertRaisesRegex(ValueError, "Train/val mirror overlap"):
            validate_rows(rows, self.cfg, check_files=False)

    def test_crop_cannot_escape_annotated_segment(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["annotation_end"] = "5"
        with self.assertRaisesRegex(ValueError, "Invalid crop"):
            validate_rows(rows, self.cfg, check_files=False)

    def test_tagged_and_nan_caption_timestamps(self):
        path = self.root / "captions.txt"
        path.write_text("a person jumps.#tokens#1.5#5.0\na person walks.#tokens#nan#0.0\n")
        caps = captions_for(path, 160, 20)
        self.assertEqual((caps[0]["start"], caps[0]["end"]), (30, 100))
        self.assertEqual((caps[1]["start"], caps[1]["end"]), (0, 160))
        self.assertTrue(caps[1]["nan_timestamp_coerced_to_zero"])

    def test_neighbors_exclude_all_same_source_samples(self):
        x = normalize(np.array([[1, 0], [1, .001], [1, .05], [0, 1], [.05, 1], [.001, 1]], dtype=np.float32))
        labels = np.array(["a", "a", "a", "b", "b", "b"])
        sources = np.array(["one", "one", "two", "three", "three", "four"])
        result = latent_metrics(x, labels, sources, 1)
        self.assertEqual(result["knn_purity"], 1.0)
        self.assertGreater(result["silhouette_cosine"], .9)
        for i, neighbors in enumerate(result["neighbors"]):
            self.assertTrue(all(sources[j] != sources[i] for j in neighbors))

    def test_zero_latent_cannot_silently_enter_cosine_metrics(self):
        with self.assertRaisesRegex(ValueError, "zero latent"):
            normalize(np.zeros((2, 512)))

    def test_paired_bootstrap_preserves_fixed_improvement(self):
        base = np.array([.1, .2, .3, .4])
        result = paired_bootstrap(base, base + .1, np.array(["a", "a", "b", "b"]), 100, 42)
        np.testing.assert_allclose(result["conditional_bootstrap_95_percentile_interval"], [.1, .1])

    def test_finalize_cannot_promote_pending_labels(self):
        csv = self.root / "pending.csv"
        write_rows(csv, self.rows)
        with self.assertRaisesRegex(ValueError, ">=2 accepted"):
            finalize(self.cfg, csv, self.root / "reviewed")


if __name__ == "__main__":
    unittest.main()
