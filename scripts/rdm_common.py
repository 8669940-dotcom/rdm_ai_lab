"""Dependency-free calculations shared by the RDM preparation tools.

All costs use the README's unit, 만원. These helpers do not fit a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "failure_within_30days"
COMMON_COLUMNS = (
    "equipment_id", "equipment_type", "unit_code", "equipment_age_years",
    "maintenance_date", "maintenance_type", "fault_code", "fault_category",
    "severity", "downtime_hours", "repair_duration_hours", "parts_replaced_count",
    "maintenance_cost", "maintenance_success", "cumulative_operating_hours",
    "days_since_last_maintenance",
)
NUMERIC_COLUMNS = (
    "equipment_age_years", "severity", "downtime_hours", "repair_duration_hours",
    "parts_replaced_count", "maintenance_cost", "maintenance_success",
    "cumulative_operating_hours", "days_since_last_maintenance", "availability_rate",
)
EXPECTED = {
    "drone_train.csv": {"rows": 2346, "start": "2023-01-01", "end": "2024-12-31",
                        "columns": (*COMMON_COLUMNS, "availability_rate", TARGET)},
    "drone_test.csv": {"rows": 1241, "start": "2025-01-01", "end": "2025-12-31",
                       "columns": (*COMMON_COLUMNS, "availability_rate")},
    "drone_this_month.csv": {"rows": 106, "start": "2026-01-01", "end": "2026-01-31",
                             "columns": COMMON_COLUMNS},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"{path.name}: missing or duplicate column names")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: CSV row width differs from the header")
    return columns, rows


def load_config() -> dict:
    config = json.loads((ROOT / "analysis_config.json").read_text(encoding="utf-8"))
    date.fromisoformat(config["report_month"] + "-01")
    date.fromisoformat(config["validation_start"])
    if type(config["capacity"]) is not int or config["capacity"] < 1:
        raise ValueError("capacity must be a positive integer")
    if config["cost_unit"] != "만원":
        raise ValueError("Costs must use the README unit: 만원")
    if config["inspection_cost"] <= 0 or any(c <= 0 for c in config["miss_costs"]):
        raise ValueError("Costs must be positive")
    threshold = config["risk_threshold"]
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("risk_threshold must be null or a probability")
    return config


def latest_record_indices(rows: list[dict[str, str]]) -> list[int]:
    """One most recent row per equipment; refuse ambiguous same-day latest rows."""
    grouped: dict[str, list[tuple[date, int]]] = {}
    for index, row in enumerate(rows):
        equipment = row["equipment_id"].strip()
        if not equipment:
            raise ValueError("equipment_id is missing")
        day = date.fromisoformat(row["maintenance_date"])
        grouped.setdefault(equipment, []).append((day, index))
    selected = []
    for equipment in sorted(grouped):
        latest_day = max(day for day, _ in grouped[equipment])
        matches = [index for day, index in grouped[equipment] if day == latest_day]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous latest maintenance: {equipment} / {latest_day}")
        selected.append(matches[0])
    return selected


def validate_probabilities(probabilities: list[float]) -> list[float]:
    values = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("prob must be finite and between 0 and 1")
    return values


def rank_latest(rows: list[dict[str, str]], probabilities: list[float],
                capacity: int = 20) -> list[dict]:
    """Rank latest rows by probability, then equipment_id for reproducible ties.

    Probabilities must be aligned with the original input rows. Return copies,
    retaining the zero-based source_index for traceability. No risk cutoff is used.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if len(rows) != len(probabilities):
        raise ValueError("Prediction count differs from source row count")
    values = validate_probabilities(probabilities)
    indices = latest_record_indices(rows)
    indices.sort(key=lambda index: (-values[index], rows[index]["equipment_id"]))
    return [dict(rows[index], prob=values[index], source_index=index, rank=rank)
            for rank, index in enumerate(indices[:capacity], start=1)]


def threshold_cost(labels: list[int], probabilities: list[float], threshold: float,
                   miss_cost: float = 300, inspection_cost: float = 60) -> dict:
    """README cost, without a capacity limit: FN * miss_cost + flags * inspection_cost."""
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Nonempty labels and probabilities must have equal lengths")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Labels must be known binary outcomes; never fill absent labels")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if any(not math.isfinite(cost) or cost <= 0 for cost in (miss_cost, inspection_cost)):
        raise ValueError("Costs must be finite and positive")
    values = validate_probabilities(probabilities)
    flags = [value >= threshold for value in values]
    missed = sum(label == 1 and not flag for label, flag in zip(labels, flags))
    detected = sum(label == 1 and flag for label, flag in zip(labels, flags))
    return {"threshold": threshold, "flagged": sum(flags), "true_positives": detected,
            "false_negatives": missed, "miss_cost": miss_cost,
            "inspection_cost": inspection_cost, "cost_unit": "만원",
            "total_cost": missed * miss_cost + sum(flags) * inspection_cost}
