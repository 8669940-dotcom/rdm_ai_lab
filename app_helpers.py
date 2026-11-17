"""Pure display calculations; the app reads trained outputs without fitting models."""
from __future__ import annotations

import math
import pandas as pd

SORT_FIELDS = {"확률": "prob", "드론": "equipment_id", "유형": "equipment_type", "부대": "unit_code",
               "최근 정비일": "maintenance_date", "정비 구분": "maintenance_type"}
GROUP_FIELDS = {"유형별": "equipment_type", "부대별": "unit_code", "정비 구분별": "maintenance_type"}
DISPLAY = {"rank": "전체 순위", "equipment_id": "드론", "equipment_type": "유형", "unit_code": "부대",
           "maintenance_date": "최근 정비일", "maintenance_type": "정비 구분", "prob": "재고장 확률",
           "selected": "선정", "risk_flag": "위험 표시", "reason": "위험 상승 요인", "recommendation": "권고"}


def plan_frame(results: dict, capacity: int, threshold: float | None) -> pd.DataFrame:
    frame = pd.DataFrame(results["current"]["equipment"]).copy()
    if type(capacity) is not int or not 1 <= capacity <= len(frame):
        raise ValueError("정원은 전체 후보 수 이내의 양의 정수여야 합니다.")
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("위험 임계값은 0~1이어야 합니다.")
    frame = frame.sort_values(["prob", "equipment_id"], ascending=[False, True]).reset_index(drop=True)
    frame["rank"] = frame.index + 1
    frame["selected"] = frame["rank"] <= capacity
    frame["risk_flag"] = frame["prob"].ge(threshold) if threshold is not None else False
    return frame


def filter_frame(frame, types=(), unit="전체", maintenance=(), query="", group=None):
    result = frame.copy()
    if types:
        result = result[result["equipment_type"].isin(types)]
    if unit != "전체":
        result = result[result["unit_code"].eq(unit)]
    if maintenance:
        result = result[result["maintenance_type"].isin(maintenance)]
    if query.strip():
        result = result[result["equipment_id"].str.contains(query.strip(), case=False, regex=False)]
    if group:
        result = result[result[group["column"]].eq(group["value"])]
    return result


def sorted_frame(frame, field="확률", ascending=False):
    column = SORT_FIELDS[field]
    columns = [column] if column == "equipment_id" else [column, "equipment_id"]
    order = [ascending] if len(columns) == 1 else [ascending, True]
    return frame.sort_values(columns, ascending=order, kind="stable").reset_index(drop=True)


def group_summary(frame, column):
    return frame.groupby(column, dropna=False, sort=True).agg(
        후보=("equipment_id", "size"), 선정=("selected", "sum"),
        위험표시=("risk_flag", "sum"), 평균확률=("prob", "mean")
    ).reset_index()


def export_csv(frame):
    columns = list(DISPLAY)
    return frame[[c for c in columns if c in frame]].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
