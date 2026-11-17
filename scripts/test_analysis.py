"""Point-in-time, cost policy, frozen artifact and display invariants."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from app_helpers import filter_frame, plan_frame, sorted_frame
from rdm_model import QuartileRepairer, TARGET, build_features, cost_sweep, evaluate_policy, load_data, metrics
from scripts.rdm_common import ROOT, sha256
from train import training_mask


def example(equipment="DR-A", day="2024-01-01", maintenance="예방정비", **changes):
    row = {"equipment_id": equipment, "equipment_type": "정찰드론", "unit_code": "UNIT-01", "equipment_age_years": 3.,
           "maintenance_date": day, "maintenance_type": maintenance, "fault_code": None, "fault_category": None,
           "severity": 0, "downtime_hours": 1., "repair_duration_hours": .5, "parts_replaced_count": 0,
           "maintenance_cost": 20., "maintenance_success": 1, "cumulative_operating_hours": 100.,
           "days_since_last_maintenance": np.nan, TARGET: 0, "_source": "train"}
    row.update(changes)
    return row


class TemporalTests(unittest.TestCase):
    def test_appended_future_and_changed_outcomes_do_not_change_past_features(self):
        records = pd.DataFrame([example(), example(day="2024-02-01", maintenance="긴급정비")])
        expected = build_features(records)
        future = pd.concat([records, pd.DataFrame([example(day="2024-03-01", maintenance="긴급정비", severity=5)])], ignore_index=True)
        future[TARGET] = 1
        assert_frame_equal(expected, build_features(future).iloc[:2])

    def test_same_day_records_do_not_count_each_other_as_history(self):
        data = pd.DataFrame([example(maintenance="긴급정비"), example()])
        features = build_features(data)
        self.assertEqual(features["prior_maintenance_count"].tolist(), [0, 0])
        self.assertEqual(features["has_prior_fault"].tolist(), [0, 0])

    def test_file_boundary_does_not_reset_history(self):
        data = pd.DataFrame([example(maintenance="긴급정비"), example(day="2025-01-01", _source="test")])
        features = build_features(data)
        self.assertEqual(features.iloc[1]["prior_maintenance_count"], 1)
        self.assertEqual(features.iloc[1]["days_since_prior_fault"], 366)

    def test_thirty_day_label_boundary_is_strict(self):
        raw = pd.DataFrame([example(day="2024-05-31"), example(day="2024-06-01")])
        raw["maintenance_date"] = pd.to_datetime(raw["maintenance_date"])
        self.assertEqual(training_mask(raw, "2024-07-01", 30).tolist(), [True, False])

    def test_quartile_repairs_use_training_bounds_and_preserve_inliers(self):
        repair = QuartileRepairer().fit(pd.DataFrame({"maintenance_cost": [10., 20., 30., 40., 50.]}))
        original = repair.policy().copy()
        transformed = repair.transform(pd.DataFrame({"maintenance_cost": [9000., 100., -100., 60., np.nan],
                                                     "downtime_hours": [999., 2., 3., 4., np.nan]}))
        self.assertEqual(repair.policy(), original)
        self.assertEqual(transformed.maintenance_cost.iloc[:4].tolist(), [40., 40., 20., 60.])
        self.assertTrue(pd.isna(transformed.maintenance_cost.iloc[4]))
        self.assertTrue(pd.isna(transformed.downtime_hours.iloc[0]))

    def test_na_features_and_blank_categories_do_not_become_valid_hours(self):
        features = build_features(pd.DataFrame([example(downtime_hours=999.)]))
        self.assertTrue(pd.isna(features.iloc[0].downtime_hours))
        self.assertEqual(features.iloc[0].fault_category, "N/A")

    def test_policy_reports_precision_recall_tpr_fpr(self):
        actual = evaluate_policy([1, 0, 1, 0], [.9, .8, .7, .1],
                                 {"policy": "threshold", "threshold": .75, "miss_cost": 300, "inspection_cost": 60})
        for key in ("precision", "recall", "tpr", "fpr"):
            self.assertEqual(actual[key], .5)
        for key in ("true_positives", "false_positives", "false_negatives", "true_negatives"):
            self.assertEqual(actual[key], 1)

    def test_cost_search_includes_no_flags_and_exact_boundary(self):
        best, table = cost_sweep([0, 0], [.5, 1.])
        self.assertEqual(best["policy"], "none")
        best, table = cost_sweep([1, 0, 1], [.8, .2, .6])
        self.assertEqual(best["threshold"], .6)
        self.assertEqual(best["cost"], 120)
        self.assertEqual(best["cost"], min(r["cost"] for r in table))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / "reports/analysis_results.json").read_text(encoding="utf-8"))

    def test_output_hashes_and_training_boundaries(self):
        for path, digest in self.result["code_sha256"].items():
            with self.subTest(code=path):
                self.assertEqual(sha256(ROOT / path), digest, "Training code/config changed: regenerate analysis")
        manifest = json.loads((ROOT / "reports/output_manifest.json").read_text(encoding="utf-8"))
        for path, digest in manifest.items():
            with self.subTest(path=path):
                self.assertEqual(sha256(ROOT / path), digest)
        for split in self.result["splits"].values():
            self.assertLess(split["latest_label_end"], split["prediction_start"])

    def test_saved_holdout_metrics_recompute(self):
        predictions = pd.read_csv(ROOT / "reports/validation_predictions_2024H2.csv")
        actual = metrics(predictions[TARGET], predictions["prob"])
        for metric in ("ap", "pr_auc_trapezoid", "roc_auc", "brier"):
            self.assertAlmostEqual(actual[metric], self.result["evaluation"]["metrics"][metric], places=12)
        self.assertIsNone(self.result["test"]["metrics"])

    def test_serialized_submission_model_reproduces_export(self):
        raw = load_data()
        features = build_features(raw)
        model = joblib.load(ROOT / "artifacts/submission_model.joblib")["model"]
        actual = model.predict_proba(features[raw["_source"].eq("test")])[:, 1]
        exported = pd.read_csv(ROOT / "reports/test_predictions_2025.csv")["prob"].to_numpy()
        np.testing.assert_allclose(actual, exported, rtol=0, atol=1e-14)

    def test_capacity_and_filtering_leave_global_rank_intact(self):
        frame = plan_frame(self.result, 20, self.result["threshold_policies"]["300"]["threshold"])
        smaller = plan_frame(self.result, 5, self.result["threshold_policies"]["300"]["threshold"])
        self.assertEqual(frame["risk_flag"].tolist(), smaller["risk_flag"].tolist())
        self.assertEqual(int(smaller["selected"].sum()), 5)
        unit = frame.iloc[0]["unit_code"]
        filtered = filter_frame(frame, unit=unit)
        self.assertEqual(filtered["rank"].tolist(), frame[frame["unit_code"].eq(unit)]["rank"].tolist())
        for field in ("확률", "드론", "유형", "부대", "최근 정비일", "정비 구분"):
            for ascending in (False, True):
                self.assertEqual(set(sorted_frame(filtered, field, ascending)["equipment_id"]), set(filtered["equipment_id"]))


if __name__ == "__main__":
    unittest.main()
