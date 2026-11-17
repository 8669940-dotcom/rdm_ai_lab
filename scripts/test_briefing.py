"""Hand-computed cases validate the independent report metric checker."""
import unittest
from build_briefing import manual_metrics


class BriefingMathTests(unittest.TestCase):
    def test_equal_scores_group_ties_and_distinguish_ap_from_trapezoid(self):
        result = manual_metrics([1, 0], [.5, .5])
        self.assertEqual(result["ap"], .5)
        self.assertEqual(result["pr_auc_trapezoid"], .75)
        self.assertEqual(result["roc_auc"], .5)
        self.assertEqual(result["brier"], .25)

    def test_perfect_binary_forecast(self):
        result = manual_metrics([0, 1], [0., 1.])
        self.assertEqual(result["ap"], 1.)
        self.assertEqual(result["roc_auc"], 1.)
        self.assertEqual(result["brier"], 0.)

    def test_reversed_forecast(self):
        result = manual_metrics([1, 0], [0., 1.])
        self.assertEqual(result["ap"], .5)
        self.assertEqual(result["pr_auc_trapezoid"], .25)
        self.assertEqual(result["roc_auc"], 0.)
        self.assertEqual(result["brier"], 1.)


if __name__ == "__main__":
    unittest.main()
