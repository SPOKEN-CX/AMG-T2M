"""Regression tests for development split isolation, source deduplication and pooling."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common import validate_rows
from expand_actions import choose_feature, collect, pool_sequence
from test_protocol import ProtocolTests


class ExpansionTests(unittest.TestCase):
    def test_validation_requires_explicit_development_purpose(self):
        fixture = ProtocolTests()
        fixture.setUp()
        try:
            cfg = copy.deepcopy(fixture.cfg)
            cfg["split"] = "val"
            with self.assertRaisesRegex(ValueError, "explicit development_diagnostics"):
                validate_rows(fixture.rows, cfg, check_files=False)
            (fixture.root / "val.txt").write_text("000003\n000004\n000005\n000006\n")
            (fixture.root / "test.txt").write_text("000010\n")
            cfg["purpose"] = "development_diagnostics"
            self.assertEqual(validate_rows(fixture.rows, cfg, check_files=False), {"a": 2, "b": 2})
        finally:
            fixture.tearDown()

    def test_temporal_pooling_retains_order(self):
        a = torch.tensor([[[0., 0., 2., 2.], [1., 1., 3., 3.]]])
        b = a.flip(-1)
        torch.testing.assert_close(pool_sequence(a, "mean"), pool_sequence(b, "mean"))
        torch.testing.assert_close(pool_sequence(a, "mean_std"), pool_sequence(b, "mean_std"))
        self.assertFalse(torch.equal(pool_sequence(a, "pyramid2"), pool_sequence(b, "pyramid2")))

    def test_larger_gap_cannot_be_selected_by_degrading_ours(self):
        scores = {
            "post_mean": {"delta_silhouette": .02, "delta_purity": .01,
                          "tmr": {"macro_silhouette": .07, "macro_neighbor_purity": .61}},
            "weaker_absolute": {"delta_silhouette": .04, "delta_purity": .08,
                                "tmr": {"macro_silhouette": .05, "macro_neighbor_purity": .58}}}
        v2 = {"primary_feature": "post_mean", "require_non_degrading_tmr": True}
        self.assertEqual(choose_feature(scores, v2), "post_mean")
        # Old saved protocols remain reproducible and are explicitly exploratory.
        self.assertEqual(choose_feature(scores, {"primary_feature": "post_mean"}), "weaker_absolute")

    def test_caption_consensus_and_original_recording_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "texts").mkdir(); (root / "new_joint_vecs").mkdir()
            (root / "val.txt").write_text("000003\n000004\n000005\n")
            for ident in ("000003", "000004", "000005"):
                np.save(root / "new_joint_vecs" / f"{ident}.npy", np.zeros((64, 263), np.float32))
                captions = ["a person claps", "someone claps"] if ident != "000005" else ["a person walks then claps", "someone claps"]
                (root / "texts" / f"{ident}.txt").write_text("\n".join(c + "#tokens#0#0" for c in captions))
            protocol = {"classes": [{"name": "clap", "pattern": r"\bclaps\b"}, {"name": "walk", "pattern": r"\bwalks\b"}],
                        "window_size": 64, "seed": 42, "minimum_matching_captions": 2}
            annotations = {"000003": {"path": "KIT/shared"}, "000004": {"path": "KIT/shared"}, "000005": {"path": "KIT/other"}}
            groups, excluded = collect("val", {"dataset_root": str(root), "fps": 20}, protocol, annotations)
            self.assertEqual(len(groups["clap"]), 1)
            self.assertEqual(excluded["duplicate_recording"], 1)
            self.assertEqual(excluded["multiple_actions"], 1)

    def test_dense_screening_excludes_invalid_caption_metadata_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "texts").mkdir()
            (root / "new_joint_vecs").mkdir()
            (root / "train.txt").write_text("000001\n000002\n")
            for ident in ("000001", "000002"):
                np.save(root / "new_joint_vecs" / f"{ident}.npy", np.zeros((64, 263), np.float32))
                stamps = "2#2" if ident == "000001" else "0#0"
                (root / "texts" / f"{ident}.txt").write_text(
                    f"a person claps#tokens#{stamps}\nsomeone claps#tokens#0#0")
            protocol = {"classes": [{"name": "clap", "pattern": r"\bclaps\b"}],
                        "window_size": 64, "seed": 42, "minimum_matching_captions": 2}
            cfg = {"dataset_root": str(root), "fps": 20}
            annotations = {"000001": {"path": "KIT/bad"}, "000002": {"path": "KIT/good"}}
            with self.assertRaisesRegex(ValueError, "Invalid caption timestamps"):
                collect("train", cfg, protocol, annotations)
            groups, excluded = collect("train", cfg, protocol, annotations, skip_invalid_captions=True)
            self.assertEqual([row["motion_id"] for row in groups["clap"]], ["000002"])
            self.assertEqual(excluded["invalid_caption_metadata"], 1)


if __name__ == "__main__":
    unittest.main()
