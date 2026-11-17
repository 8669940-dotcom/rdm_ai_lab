"""Observable dashboard behavior in the real Streamlit script runtime."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    entrypoint = "app.py"

    def app(self):
        app = AppTest.from_file(str(ROOT / self.entrypoint), default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def metric(self, app, label):
        return next(item.value for item in app.metric if item.label == label)

    def test_default_plan_and_capacity_scenario(self):
        app = self.app()
        self.assertEqual(self.metric(app, "재점검 선정"), "20대")
        risk = self.metric(app, "위험 기준 이상")
        app.number_input(key="capacity").set_value(5).run()
        self.assertFalse(app.exception)
        self.assertEqual(self.metric(app, "재점검 선정"), "5대")
        self.assertEqual(self.metric(app, "위험 기준 이상"), risk)

    def test_manual_threshold_does_not_change_capacity(self):
        app = self.app()
        app.toggle(key="manual_threshold").set_value(True).run()
        app.slider(key="threshold_percent").set_value(100.).run()
        self.assertFalse(app.exception)
        self.assertEqual(self.metric(app, "위험 기준 이상"), "0대")
        self.assertEqual(self.metric(app, "재점검 선정"), "20대")

    def test_empty_filter_is_visible_and_keeps_plan_count(self):
        app = self.app()
        app.text_input(key="search").set_value("NO-SUCH-DRONE").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("조건에 맞는" in item.value for item in app.info))
        self.assertEqual(self.metric(app, "재점검 선정"), "20대")

    def test_other_views_render(self):
        app = self.app()
        for page in ("모델 검증", "데이터·과정", "재점검 계획"):
            app.segmented_control(key="view").set_value(page).run()
            self.assertFalse(app.exception)

    def test_reanalysis_comparison_is_korean_and_has_twelve_configurations(self):
        app = self.app()
        app.segmented_control(key="view").set_value("모델 검증").run()
        tables = [x.value for x in app.dataframe if "모델·피처 구성" in x.value.columns]
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0]), 12)
        self.assertTrue(tables[0]["모델·피처 구성"].str.contains("현재 정비·상태").all())
        self.assertFalse(tables[0]["모델·피처 구성"].str.contains("_quartile").any())

    def test_group_back_restores_scope_and_sort(self):
        app = self.app()
        app.selectbox(key="sort_field").set_value("드론").run()
        app.session_state["group_filter"] = {"column": "equipment_type", "value": "정찰드론"}
        app.session_state["return_scope"] = "선정 대상"
        app.run()
        self.assertFalse(app.exception)
        app.button(key="back_group").click().run()
        self.assertFalse(app.exception)
        self.assertIsNone(app.session_state["group_filter"])
        self.assertEqual(app.selectbox(key="sort_field").value, "드론")
        self.assertEqual(app.segmented_control(key="list_scope").value, "선정 대상")


class StyledDashboardTests(DashboardTests):
    """The styled entry point must preserve the same six observable contracts."""
    entrypoint = "app_styled.py"


if __name__ == "__main__":
    unittest.main()
