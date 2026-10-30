"""Fair, reproducible sampling for the moderate-size presentation views."""
import unittest

from optimize_display import select_indices


class DisplaySelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"sample_id": f"{label}_{i:03d}", "label": label, "diagnostic_score": i}
                     for label, count in (("a", 80), ("b", 35), ("c", 70)) for i in range(count)]

    def test_cap_keeps_rare_class_and_is_independent_of_order_and_scores(self):
        original = select_indices(self.rows, ["a", "b", "c"], 60, 20261029)
        selected = {self.rows[i]["sample_id"] for i in original}
        changed = [{**row, "diagnostic_score": -100*row["diagnostic_score"]} for row in reversed(self.rows)]
        reordered = select_indices(changed, ["a", "b", "c"], 60, 20261029)
        self.assertEqual(selected, {changed[i]["sample_id"] for i in reordered})
        self.assertEqual(sum(self.rows[i]["label"] == "b" for i in original), 35)
        self.assertEqual(len(original), 155)

    def test_balanced_view_rejects_invented_extra_samples(self):
        with self.assertRaisesRegex(ValueError, "Insufficient unique samples"):
            select_indices(self.rows, ["a", "b"], 40, 20261029, balanced=True)

    def test_focus_samples_are_nested_in_the_same_overview(self):
        overview = set(select_indices(self.rows, ["a", "b", "c"], 60, 20261029))
        focus = set(select_indices(self.rows, ["a", "c"], 40, 20261029, balanced=True))
        self.assertTrue(focus <= overview)
        self.assertEqual(len(focus), 80)


if __name__ == "__main__":
    unittest.main()
