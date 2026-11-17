"""Point-in-time features, reproducible model candidates and analytical helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.rdm_common import EXPECTED, ROOT, TARGET

BASE_NUMERIC = ["equipment_age_years", "severity", "downtime_hours", "repair_duration_hours",
                "parts_replaced_count", "maintenance_cost", "maintenance_success",
                "cumulative_operating_hours", "days_since_last_maintenance", "gap_missing",
                "month_sin", "month_cos"]
HISTORY_NUMERIC = ["prior_maintenance_count", "prior_maintenance_90d", "prior_faults_90d",
                   "days_since_prior_fault", "has_prior_fault", "flight_hours_since_preventive",
                   "has_prior_preventive", "prior_unsuccessful_180d"]
BASE_CATEGORICAL = ["equipment_type", "maintenance_type", "fault_category"]
FEATURE_LABELS = {
    "equipment_age_years": "경과연수", "severity": "고장 심각도", "downtime_hours": "운용중단 시간",
    "repair_duration_hours": "수리 시간", "parts_replaced_count": "교체 부품 수", "maintenance_cost": "정비 비용",
    "maintenance_success": "정비 성공 여부", "cumulative_operating_hours": "누적 비행시간",
    "days_since_last_maintenance": "직전 정비 경과일", "gap_missing": "직전 정비 이력 공란",
    "month_sin": "정비 월 주기(sin)", "month_cos": "정비 월 주기(cos)",
    "prior_maintenance_count": "이전 관측 정비 수", "prior_maintenance_90d": "이전 90일 정비 수",
    "prior_faults_90d": "이전 90일 사후·긴급정비 수", "days_since_prior_fault": "이전 관측 고장 경과일",
    "has_prior_fault": "이전 관측 고장 존재", "flight_hours_since_preventive": "이전 예방정비 후 비행시간 증가",
    "has_prior_preventive": "이전 예방정비 관측", "prior_unsuccessful_180d": "이전 180일 정비 실패 수",
    "equipment_type": "드론 유형", "maintenance_type": "정비 구분", "fault_category": "고장 유형",
    "unit_code": "부대", "fault_code": "고장 코드",
}
CANDIDATES = [
    {"key": "lr_basic_cap", "kind": "logistic", "history": False, "context": False, "cap": True},
    {"key": "lr_history_raw", "kind": "logistic", "history": True, "context": False, "cap": False},
    {"key": "lr_history_cap", "kind": "logistic", "history": True, "context": False, "cap": True},
    {"key": "lr_context_cap", "kind": "logistic", "history": True, "context": True, "cap": True},
    {"key": "rf_history_cap", "kind": "random_forest", "history": True, "context": False, "cap": True},
    {"key": "histgb_history_cap", "kind": "hist_gradient_boosting", "history": True, "context": False, "cap": True},
    {"key": "lgbm_history_cap", "kind": "lightgbm", "history": True, "context": False, "cap": True},
    {"key": "lgbm_context_cap", "kind": "lightgbm", "history": True, "context": True, "cap": True},
]


def load_data() -> pd.DataFrame:
    frames = []
    for filename, source in zip(EXPECTED, ("train", "test", "month")):
        frame = pd.read_csv(ROOT / "data" / filename, encoding="utf-8-sig")
        frame["maintenance_date"] = pd.to_datetime(frame["maintenance_date"], format="%Y-%m-%d")
        frame["_source"] = source
        frame["_source_row"] = np.arange(len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Ignore target/availability; every historical contributor has date < current date.

    Input and returned index/order are identical. Equipment/date ties never contribute
    to each other's history. File boundaries do not reset an equipment's history.
    """
    frame = raw.copy()
    frame["maintenance_date"] = pd.to_datetime(frame["maintenance_date"])
    output = frame[BASE_NUMERIC[:9] + BASE_CATEGORICAL + ["unit_code", "fault_code"]].copy()
    for column in ("fault_code", "fault_category"):
        absent = np.where(frame["maintenance_type"].eq("예방정비"), "해당 없음", "미기재")
        output[column] = frame[column].where(frame[column].notna(), pd.Series(absent, index=frame.index))
    output["gap_missing"] = frame["days_since_last_maintenance"].isna().astype(int)
    month = frame["maintenance_date"].dt.month
    output["month_sin"] = np.sin(2 * np.pi * month / 12)
    output["month_cos"] = np.cos(2 * np.pi * month / 12)
    histories = {}
    for _, group in frame.groupby("equipment_id", sort=True):
        previous = []
        for index, row in group.sort_values("maintenance_date", kind="stable").iterrows():
            day = row["maintenance_date"]
            past = [old for old in previous if old["maintenance_date"] < day]
            last90 = [old for old in past if old["maintenance_date"] >= day - pd.Timedelta(days=90)]
            last180 = [old for old in past if old["maintenance_date"] >= day - pd.Timedelta(days=180)]
            faults = [old for old in past if old["maintenance_type"] in ("사후정비", "긴급정비")]
            preventive = [old for old in past if old["maintenance_type"] == "예방정비"]
            histories[index] = {
                "prior_maintenance_count": len(past), "prior_maintenance_90d": len(last90),
                "prior_faults_90d": sum(old["maintenance_type"] in ("사후정비", "긴급정비") for old in last90),
                "days_since_prior_fault": (day - faults[-1]["maintenance_date"]).days if faults else np.nan,
                "has_prior_fault": int(bool(faults)),
                "flight_hours_since_preventive": (max(0., row["cumulative_operating_hours"] - preventive[-1]["cumulative_operating_hours"])
                                                  if preventive else np.nan),
                "has_prior_preventive": int(bool(preventive)),
                "prior_unsuccessful_180d": sum(old["maintenance_success"] == 0 for old in last180),
            }
            previous.append(row)
    return output.join(pd.DataFrame.from_dict(histories, orient="index"))


class QuantileCapper(TransformerMixin, BaseEstimator):
    """Learn 99th percentile bounds for time/cost from fit data only; no row removal."""
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.upper_ = {c: float(X[c].quantile(.99)) for c in ("downtime_hours", "maintenance_cost") if c in X}
        return self

    def transform(self, X):
        result = X.copy()
        if self.enabled:
            for column, upper in self.upper_.items():
                result[column] = result[column].clip(upper=upper)
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)


def feature_columns(spec: dict) -> tuple[list[str], list[str]]:
    return (BASE_NUMERIC + (HISTORY_NUMERIC if spec["history"] else []),
            BASE_CATEGORICAL + (["unit_code", "fault_code"] if spec["context"] else []))


def make_model(spec: dict, seed: int = 42) -> Pipeline:
    numeric, categorical = feature_columns(spec)
    prep = ColumnTransformer([
        ("numeric", Pipeline([("cap", QuantileCapper(spec["cap"])),
                              ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="미기재")),
                               ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop").set_output(transform="pandas")
    estimators = {
        "logistic": lambda: LogisticRegression(C=.5, max_iter=2000, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=240, max_depth=6, min_samples_leaf=8,
                                                        n_jobs=2, random_state=seed),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=160, learning_rate=.04,
                                                                          max_leaf_nodes=15, l2_regularization=2,
                                                                          early_stopping=False, random_state=seed),
        "lightgbm": lambda: LGBMClassifier(n_estimators=180, learning_rate=.035, num_leaves=15,
                                            min_child_samples=20, reg_lambda=2., n_jobs=2,
                                            random_state=seed, verbosity=-1, deterministic=True, force_col_wise=True),
    }
    return Pipeline([("prep", prep), ("model", estimators[spec["kind"]]())])


def metrics(y, probabilities) -> dict:
    y, probabilities = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    both_classes = len(np.unique(y)) == 2
    precision, recall, _ = precision_recall_curve(y, probabilities)
    return {"rows": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "ap": float(average_precision_score(y, probabilities)) if y.sum() else None,
            "pr_auc_trapezoid": float(auc(recall, precision)) if y.sum() else None,
            "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
            "brier": float(brier_score_loss(y, probabilities))}


def monthly_topk(raw: pd.DataFrame, probabilities, capacity=20, seed=42, repeats=1000) -> list[dict]:
    frame = raw.assign(prob=np.asarray(probabilities)).copy()
    frame["month"] = frame["maintenance_date"].dt.strftime("%Y-%m")
    output = []
    # Stable independent seed for each month/unit makes model comparisons share baseline draws.
    for month, group in frame.groupby("month", sort=True):
        month = str(month)
        for unit in ("latest_drone", "maintenance_record"):
            pool = group.sort_values(["maintenance_date", "equipment_id"], kind="stable")
            if unit == "latest_drone":
                pool = pool.drop_duplicates("equipment_id", keep="last")
            pool = pool.sort_values(["equipment_id", "maintenance_date"], kind="stable")
            k = min(capacity, len(pool))
            selected = pool.sort_values(["prob", "equipment_id", "maintenance_date"], ascending=[False, True, False]).head(k)
            aged = pool.sort_values(["equipment_age_years", "equipment_id", "maintenance_date"], ascending=[False, True, False]).head(k)
            labels = pool[TARGET].to_numpy(dtype=int)
            rng = np.random.default_rng(seed + int(month.replace("-", "")) + (unit == "maintenance_record"))
            random_hits = np.array([labels[rng.choice(len(pool), k, replace=False)].sum() for _ in range(repeats)])
            output.append({"month": month, "unit": unit, "candidates": len(pool), "selected": k,
                           "positives": int(labels.sum()), "model_hits": int(selected[TARGET].sum()),
                           "age_hits": int(aged[TARGET].sum()), "random_hits_mean": float(random_hits.mean()),
                           "random_hits_p025": float(np.quantile(random_hits, .025)),
                           "random_hits_p975": float(np.quantile(random_hits, .975)),
                           "random_expected_hits": float(k * labels.mean()), "random_repeats": repeats,
                           "model_cost_300": float((labels.sum() - selected[TARGET].sum()) * 300 + k * 60),
                           "age_cost_300": float((labels.sum() - aged[TARGET].sum()) * 300 + k * 60)})
    return output


def cost_sweep(y, probabilities, miss_cost=300, inspection_cost=60) -> tuple[dict, list[dict]]:
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    values = np.unique(np.r_[0., p, 1.])
    table: list[dict] = []
    for threshold in values:
        flags = p >= threshold
        fn = int(y[~flags].sum())
        table.append({"threshold": float(threshold), "policy": "threshold", "flagged": int(flags.sum()),
                      "true_positives": int(y[flags].sum()), "false_negatives": fn,
                      "cost": fn * miss_cost + int(flags.sum()) * inspection_cost})
    table.append({"threshold": None, "policy": "none", "flagged": 0, "true_positives": 0,
                  "false_negatives": int(y.sum()), "cost": int(y.sum()) * miss_cost})
    best = min(table, key=lambda r: (r["cost"], r["flagged"], -(r["threshold"] if r["threshold"] is not None else 2)))
    return {**best, "miss_cost": miss_cost, "inspection_cost": inspection_cost, "unit": "maintenance_record"}, table


def evaluate_policy(y, probabilities, policy: dict) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    flags = p >= policy["threshold"] if policy["policy"] == "threshold" else np.zeros(len(p), dtype=bool)
    fn, tp = int(y[~flags].sum()), int(y[flags].sum())
    return {"flagged": int(flags.sum()), "true_positives": tp, "false_negatives": fn,
            "cost": fn * policy["miss_cost"] + int(flags.sum()) * policy["inspection_cost"],
            "precision": tp / int(flags.sum()) if flags.sum() else None,
            "recall": tp / int(y.sum()) if y.sum() else None}
