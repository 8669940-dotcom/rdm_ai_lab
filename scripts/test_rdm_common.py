"""Small analytical invariants, using hand-calculated examples, not model scores."""
import math
import csv
import tempfile
import unittest
from pathlib import Path

from check import check_predictions
from rdm_common import ROOT, latest_record_indices, rank_latest, read_csv, threshold_cost


def record(equipment, day):
    return {"equipment_id": equipment, "maintenance_date": day}


class CalculationTests(unittest.TestCase):
    def test_latest_record_precedes_probability_ranking(self):
        rows = [record("DR-A", "2026-01-01"), record("DR-B", "2026-01-10"),
                record("DR-A", "2026-01-20")]
        selected = rank_latest(rows, [.99, .7, .2], capacity=2)
        self.assertEqual([r["equipment_id"] for r in selected], ["DR-B", "DR-A"])
        self.assertEqual([r["prob"] for r in selected], [.7, .2])
        self.assertEqual([r["source_index"] for r in selected], [1, 2])

    def test_ties_have_reproducible_equipment_order(self):
        rows = [record("DR-B", "2026-01-10"), record("DR-A", "2026-01-10")]
        self.assertEqual(rank_latest(rows, [.5, .5], 1)[0]["equipment_id"], "DR-A")

    def test_ambiguous_same_day_latest_record_is_not_silently_chosen(self):
        rows = [record("DR-A", "2026-01-10"), record("DR-A", "2026-01-10")]
        with self.assertRaises(ValueError):
            latest_record_indices(rows)

    def test_cost_uses_inclusive_threshold_and_both_cost_assumptions(self):
        # Three flags (including .2), two hits, one missed positive.
        labels, probabilities = [1, 0, 1, 1], [.2, .7, .1, .9]
        result = threshold_cost(labels, probabilities, .2)
        self.assertEqual((result["flagged"], result["true_positives"], result["false_negatives"]), (3, 2, 1))
        self.assertEqual(result["total_cost"], 480)
        self.assertEqual(threshold_cost(labels, probabilities, .2, miss_cost=600)["total_cost"], 780)

    def test_capacity_does_not_change_number_of_risk_flags(self):
        rows = [record(f"DR-{i}", "2026-01-10") for i in range(3)]
        self.assertEqual(len(rank_latest(rows, [.9, .8, .7], capacity=1)), 1)
        self.assertEqual(threshold_cost([1, 0, 1], [.9, .8, .7], .5)["flagged"], 3)

    def test_invalid_probabilities_and_absent_labels_fail(self):
        rows = [record("DR-A", "2026-01-10")]
        for value in (math.nan, math.inf, -.1, 1.1):
            with self.subTest(prob=value), self.assertRaises(ValueError):
                rank_latest(rows, [value])
        with self.assertRaises(ValueError):
            rank_latest(rows, [])
        with self.assertRaises(ValueError):
            threshold_cost([None], [.5], .3)

    def test_selection_preserves_source_and_handles_small_candidate_pool(self):
        rows = [record("DR-A", "2026-01-10")]
        before = [dict(r) for r in rows]
        selected = rank_latest(rows, [.3], capacity=20)
        self.assertEqual(len(selected), 1)
        self.assertEqual(rows, before)


class SubmissionContractTests(unittest.TestCase):
    def setUp(self):
        _, source = read_csv(ROOT / "data" / "drone_test.csv")
        self.predictions = [{"equipment_id": row["equipment_id"],
                             "maintenance_date": row["maintenance_date"], "prob": ".5"}
                            for row in source]
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "contract_fixture.csv"

    def write_predictions(self):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["equipment_id", "maintenance_date", "prob"])
            writer.writeheader()
            writer.writerows(self.predictions)

    def test_complete_key_set_can_be_reordered(self):
        # Dummy scores exist only in a temporary contract fixture, not a submission.
        self.predictions.reverse()
        self.write_predictions()
        check_predictions(self.path)

    def test_missing_source_record_fails(self):
        self.predictions.pop()
        self.write_predictions()
        with self.assertRaises(ValueError):
            check_predictions(self.path)

    def test_duplicate_key_cannot_replace_another_record(self):
        self.predictions[1] = dict(self.predictions[0])
        self.write_predictions()
        with self.assertRaises(ValueError):
            check_predictions(self.path)

    def test_nonfinite_submission_probability_fails(self):
        self.predictions[0]["prob"] = "nan"
        self.write_predictions()
        with self.assertRaises(ValueError):
            check_predictions(self.path)


if __name__ == "__main__":
    unittest.main()
