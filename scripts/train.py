"""Run chronological model selection, fixed internal evaluation and honest test export."""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve
from threadpoolctl import threadpool_limits

from rdm_model import (CANDIDATES, FEATURE_LABELS, TARGET, build_features, cost_sweep,
                       TIME_SENTINELS, evaluate_policy, feature_columns, load_data, make_model, metrics, monthly_topk)
from scripts.rdm_common import ROOT, load_config, rank_latest, sha256


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def training_mask(raw, cutoff, horizon):
    return raw["_source"].eq("train") & (raw["maintenance_date"] + pd.Timedelta(days=horizon) < pd.Timestamp(cutoff))


def fit_for_cutoff(raw, features, cutoff, spec, config):
    mask = training_mask(raw, cutoff, config["label_horizon_days"])
    model = make_model(spec, config["random_seed"])
    model.fit(features.loc[mask], raw.loc[mask, TARGET].astype(int))
    return model, mask


def explain_reference(model, X, training_features, columns):
    """One feature replaced at a time; probability sensitivity, not SHAP or causality."""
    reference = {}
    for column in columns:
        series = training_features[column]
        reference[column] = float(series.median()) if pd.api.types.is_numeric_dtype(series) else str(series.mode().iloc[0])
    original = model.predict_proba(X)[:, 1]
    changes = {}
    for column in columns:
        replaced = X.copy()
        replaced[column] = reference[column]
        changes[column] = original - model.predict_proba(replaced)[:, 1]
    result = []
    for offset, (_, row) in enumerate(X.iterrows()):
        positive = sorted([(column, float(values[offset])) for column, values in changes.items() if values[offset] > 1e-9],
                          key=lambda pair: (-pair[1], pair[0]))[:3]
        details = [{"feature": c, "label": FEATURE_LABELS.get(c, c), "value": clean(row[c]),
                    "reference": clean(reference[c]), "delta_probability": delta} for c, delta in positive]
        text = "; ".join(f"{r['label']} {r['value']} (참조값 치환 대비 +{r['delta_probability']*100:.1f}%p)" for r in details)
        result.append({"factors": details, "reason": text or "참조값 대비 뚜렷한 단일 상승 요인 없음"})
    return result, reference


def cleaning_evidence(model, raw, features, mask, role, audit_rows):
    numeric = model.named_steps["prep"].named_transformers_["numeric"]
    repair = numeric.named_steps["repair"]
    columns = list(repair.feature_names_in_)
    part = raw.loc[mask]
    original = features.loc[mask, columns]
    cleaned = repair.transform(original)
    medians = dict(zip(columns, numeric.named_steps["impute"].statistics_))
    cost = part["maintenance_cost"]
    outside = cost.lt(repair.lower_) | cost.gt(repair.upper_)
    hundredfold = outside & cost.gt(repair.error_factor * repair.q3_)
    for index in part.index:
        for field in [*TIME_SENTINELS, "maintenance_cost", "days_since_last_maintenance"]:
            value, replacement = part.loc[index, field], cleaned.loc[index, field]
            missing = pd.isna(value) or (field in TIME_SENTINELS and value == 999)
            changed = field == "maintenance_cost" and bool(outside.loc[index])
            if missing or changed:
                reason = "공란→N/A" if pd.isna(value) else "999시간→N/A" if missing else (
                    "100배 입력 오류 가정÷100 후 Q1~Q3 제한" if hundredfold.loc[index] else "IQR 후보 Q1~Q3 제한")
                audit_rows.append({"role": role, "source": part.loc[index, "_source"], "source_row": int(part.loc[index, "_source_row"]),
                                   "equipment_id": part.loc[index, "equipment_id"], "maintenance_date": part.loc[index, "maintenance_date"],
                                   "field": field, "label": FEATURE_LABELS[field], "original": value, "cleaned": replacement,
                                   "model_imputed": medians[field] if pd.isna(replacement) else replacement, "reason": reason})
    return {"rows": len(part), "time_999_to_na": {c: int(part[c].eq(999).sum()) for c in TIME_SENTINELS},
            "numeric_blank": {c: int(part[c].isna().sum()) for c in columns if c in part and part[c].isna().any()},
            "cost_outliers": int(outside.sum()), "cost_hundredfold_assumed": int(hundredfold.sum()),
            "cost_quartile_only": int((outside & ~hundredfold).sum())}


def main():
    started = time.perf_counter()
    config = load_config()
    reports, artifacts = ROOT / "reports", ROOT / "artifacts"
    reports.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    manifest = json.loads((reports / "data_manifest.json").read_text(encoding="utf-8"))
    for relative, digest in manifest.items():
        if relative.startswith("data/") and sha256(ROOT / relative) != digest:
            raise ValueError(f"Original changed: {relative}")
    raw = load_data()
    X = build_features(raw)
    select_mask = raw["_source"].eq("train") & raw["maintenance_date"].between(config["selection_start"], config["selection_end"])
    eval_mask = raw["_source"].eq("train") & raw["maintenance_date"].ge(config["validation_start"])
    selection = raw.loc[select_mask]
    evaluation = raw.loc[eval_mask]
    y_select = selection[TARGET].astype(int)
    y_eval = evaluation[TARGET].astype(int)
    comparisons, selection_predictions, selection_models = [], {}, {}
    for spec in CANDIDATES:
        tick = time.perf_counter()
        model, select_train_mask = fit_for_cutoff(raw, X, config["selection_start"], spec, config)
        p = model.predict_proba(X.loc[select_mask])[:, 1]
        result = metrics(y_select, p)
        monthly = monthly_topk(selection, p, config["capacity"], config["random_seed"])
        item = {"key": spec["key"], "label": spec["label"], "feature_group": spec["feature_group"], "spec": spec, **result,
                "latest_top20_hits": sum(r["model_hits"] for r in monthly if r["unit"] == "latest_drone"),
                "fit_predict_seconds": time.perf_counter() - tick}
        comparisons.append(item)
        selection_predictions[spec["key"]] = p
        selection_models[spec["key"]] = model
        print(f"{spec['key']}: development AP={result['ap']:.4f}, top20 hits={item['latest_top20_hits']}", flush=True)
    # Only the earlier development window selects a model. Later evaluation is not used to reselect.
    chosen = sorted(comparisons, key=lambda r: (-r["ap"], -r["latest_top20_hits"], r["brier"], r["key"]))[0]
    spec = chosen["spec"]
    p_select = selection_predictions[chosen["key"]]
    policies, cost_tables = {}, []
    for miss_cost in config["miss_costs"]:
        best, table = cost_sweep(y_select, p_select, miss_cost, config["inspection_cost"])
        policies[str(miss_cost)] = {**best, **evaluate_policy(y_select, p_select, best)}
        cost_tables.extend({**r, "miss_cost": miss_cost} for r in table)
    eval_model, eval_train_mask = fit_for_cutoff(raw, X, config["validation_start"], spec, config)
    p_eval = eval_model.predict_proba(X.loc[eval_mask])[:, 1]
    eval_metrics = metrics(y_eval, p_eval)
    eval_monthly = monthly_topk(evaluation, p_eval, config["capacity"], config["random_seed"])
    dev_monthly = monthly_topk(selection, p_select, config["capacity"], config["random_seed"])
    holdout_costs = {cost: evaluate_policy(y_eval, p_eval, policy) for cost, policy in policies.items()}
    print(f"Selected {chosen['key']}; fixed later-window evaluation AP={eval_metrics['ap']:.4f}", flush=True)

    # Fixed model at the beginning of test year; never train on immature Dec 2024 labels.
    test_model, test_train_mask = fit_for_cutoff(raw, X, "2025-01-01", spec, config)
    month_cutoff = config["report_month"] + "-01"
    month_model, month_train_mask = fit_for_cutoff(raw, X, month_cutoff, spec, config)
    test_mask, month_mask = raw["_source"].eq("test"), raw["_source"].eq("month")
    p_test = test_model.predict_proba(X.loc[test_mask])[:, 1]
    p_month = month_model.predict_proba(X.loc[month_mask])[:, 1]
    test_export = raw.loc[test_mask, ["equipment_id", "maintenance_date"]].copy()
    test_export["maintenance_date"] = test_export["maintenance_date"].dt.strftime("%Y-%m-%d")
    test_export["prob"] = p_test
    test_export.to_csv(reports / "test_predictions_2025.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    monthly_raw = raw.loc[month_mask].copy()
    month_rows = monthly_raw.assign(maintenance_date=monthly_raw["maintenance_date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    ranked = rank_latest(month_rows, p_month.tolist(), len(month_rows))
    all_indices = [monthly_raw.index[int(row["source_index"])] for row in ranked]
    numeric, categorical = feature_columns(spec)
    columns = numeric + categorical
    explanations, reference = explain_reference(month_model, X.loc[all_indices], X.loc[month_train_mask], columns)
    baseline_policy = policies[str(config["miss_costs"][0])]
    current = []
    for rank, (row, explanation) in enumerate(zip(ranked, explanations), 1):
        risk = bool(row["prob"] >= baseline_policy["threshold"]) if baseline_policy["policy"] == "threshold" else False
        faults = row.get("fault_category")
        recommendation = {"배터리": "배터리 상태·충방전 이력과 셀 편차 재점검", "모터": "모터 구동·진동·회전부 재점검",
                          "통신": "통신 연결·안테나·링크 안정성 재점검", "기체": "기체 손상·체결 상태 재점검",
                          "센서": "센서 교정·측정 일관성 재점검"}.get(faults, "정비 결과 확인 후 비행 전 기능·안전 점검")
        current.append({**row, **explanation, "selected": rank <= config["capacity"],
                        "risk_flag": risk, "recommendation": recommendation})
    current_frame = pd.DataFrame(current).drop(columns=[TARGET, "availability_rate", "factors"], errors="ignore")
    current_frame.to_csv(reports / "current_candidates_2026-01.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    current_frame.loc[current_frame["selected"]].to_csv(reports / "reinspection_plan_2026-01.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    eval_predictions = evaluation[["equipment_id", "maintenance_date", "equipment_age_years", TARGET]].copy()
    eval_predictions["prob"] = p_eval
    eval_predictions.to_csv(reports / "validation_predictions_2024H2.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(eval_monthly).to_csv(reports / "monthly_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cost_tables).to_csv(reports / "threshold_costs.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    importance = permutation_importance(eval_model, X.loc[eval_mask], y_eval, scoring="average_precision",
                                        n_repeats=5, random_state=config["random_seed"], n_jobs=1)
    importances = sorted([{"feature": column, "label": FEATURE_LABELS.get(column, column),
                           "ap_decrease_mean": float(importance.importances_mean[index]),
                           "ap_decrease_std": float(importance.importances_std[index])}
                          for index, column in enumerate(X.columns) if column in columns],
                         key=lambda r: -r["ap_decrease_mean"])
    precision, recall, _ = precision_recall_curve(y_eval, p_eval)
    fpr, tpr, _ = roc_curve(y_eval, p_eval)
    fraction_positive, mean_predicted = calibration_curve(y_eval, p_eval, n_bins=5, strategy="quantile")
    joblib.dump({"model": month_model, "spec": spec, "reference": reference}, artifacts / "rdm_model.joblib", compress=3)
    joblib.dump({"model": test_model, "spec": spec}, artifacts / "submission_model.joblib", compress=3)
    development_model = selection_models[chosen["key"]]
    audit_rows, applications, fitted = [], {}, {}
    for role, fitted_model, fit_mask, predict_mask in [
        ("development", development_model, select_train_mask, select_mask),
        ("evaluation", eval_model, eval_train_mask, eval_mask),
        ("submission", test_model, test_train_mask, test_mask),
        ("operation", month_model, month_train_mask, month_mask)]:
        fitted[role] = fitted_model.named_steps["prep"].named_transformers_["numeric"].named_steps["repair"].policy()
        applications[role + "_train"] = cleaning_evidence(fitted_model, raw, X, fit_mask, role + "_train", audit_rows)
        applications[role + "_predict"] = cleaning_evidence(fitted_model, raw, X, predict_mask, role + "_predict", audit_rows)
    pd.DataFrame(audit_rows).to_csv(reports / "cleaning_audit.csv", index=False, encoding="utf-8-sig", na_rep="N/A", float_format="%.17g")
    splits = {}
    for name, mask, cutoff in [("development_train", select_train_mask, config["selection_start"]),
                               ("evaluation_train", eval_train_mask, config["validation_start"]),
                               ("submission_train", test_train_mask, "2025-01-01"),
                               ("operation_train", month_train_mask, month_cutoff)]:
        part = raw.loc[mask]
        splits[name] = {"rows": len(part), "start": part["maintenance_date"].min(),
                        "end": part["maintenance_date"].max(), "prediction_start": cutoff,
                        "latest_label_end": part["maintenance_date"].max() + pd.Timedelta(days=config["label_horizon_days"])}
    evidence = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "data_sha256": {k: v for k, v in manifest.items() if k.startswith("data/")},
        "code_sha256": {p: sha256(ROOT / p) for p in ("rdm_model.py", "scripts/train.py", "analysis_config.json")},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__, "streamlit": streamlit.__version__},
        "selection_rule": "Development AP descending, then latest-drone top20 hits descending, Brier ascending, key ascending. No selection on 2024H2.",
        "selected_model": chosen["key"], "selected_model_label": spec["label"], "selected_spec": spec, "comparisons": comparisons,
        "splits": splits,
        "development": {"start": config["selection_start"], "end": config["selection_end"],
                        "metrics": metrics(y_select, p_select), "monthly": dev_monthly},
        "evaluation": {"start": config["validation_start"], "end": "2024-12-31", "metrics": eval_metrics,
                       "monthly": eval_monthly, "frozen_policy_costs": holdout_costs,
                       "pr_curve": [{"recall": float(r), "precision": float(p)} for r, p in zip(recall, precision)],
                       "roc_curve": [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)],
                       "calibration": [{"mean_predicted": float(p), "observed_rate": float(y)} for p, y in zip(mean_predicted, fraction_positive)]},
        "threshold_policies": policies, "threshold_tie_rule": "Minimum cost; then fewer flags; then higher threshold. Include no-flags policy.",
        "processing": {"removed_rows": 0, "policy_version": "na999_cost100_iqr_quartile_v2",
                       "fitted_by_role": fitted, "applications": applications, "audit_file": "reports/cleaning_audit.csv",
                       "numeric_imputation": "Median and missing indicators fitted only on each training period",
                       "categorical_imputation": "All blank categories are N/A; blank fault fields occur in preventive records",
                       "excluded": {"equipment_id": "Identifier for history and submission; excluded from input",
                                    "maintenance_date": "Ordering and calendar features only",
                                    "availability_rate": "Monthly end value unavailable after maintenance; absent in January input",
                                    TARGET: "Outcome, never a predictor",
                                    **({"unit_code": "Context-feature alternative lost on development AP", "fault_code": "Context-feature alternative lost on development AP"} if not spec["context"] else {})},
                       "feature_columns": columns, "feature_labels": [FEATURE_LABELS[c] for c in columns]},
        "permutation_importance": importances,
        "explanation_method": "One feature replaced by operation-training median/mode; original probability minus replaced probability. Not SHAP, additive attribution or causal effect; correlated/redundant features can mask or duplicate effects.",
        "current": {"month": config["report_month"], "as_of": "2026-01-31", "records": len(monthly_raw),
                    "candidates": len(current), "capacity": config["capacity"], "selected": min(config["capacity"], len(current)),
                    "risk_flagged": sum(1 for r in current if r["risk_flag"]),
                    "risk_outside_capacity": sum(1 for r in current if r["risk_flag"] and not r["selected"]),
                    "equipment": current},
        "test": {"rows": len(test_export), "labels_available": False, "metrics": None,
                 "prediction_file": "reports/test_predictions_2025.csv", "model_training": "submission_train"},
        "limitations": ["Educational synthetic data; performance is not evidence of operational safety or actual savings.",
                        "The later 2024 window was already inspected in earlier iterations; it is not fresh external validation of the revised workflow.",
                        "999-as-missing and hundredfold cost errors are user-directed assumptions, not verified data-generation rules. IQR repairs can suppress real high-cost events.",
                        "Instructor test labels and grading unit are unavailable; no 2025 performance is reported.",
                        "Refitting changes probability scale; the frozen development threshold may drift and needs prospective calibration.",
                        "Monthly latest-record snapshots are retrospective month-end plans, not live daily dispatch decisions.",
                        "No actual intervention outcome; cost comparison assumes selected re-failures could be prevented.",
                        "Repeated equipment are present in chronological splits; transfer to unseen equipment/units is untested."],
        "seconds": time.perf_counter() - started,
    }
    save_json(reports / "analysis_results.json", evidence)
    save_json(artifacts / "model_card.json", {k: evidence[k] for k in ("created_at_utc", "selected_model", "selected_model_label", "selected_spec", "environment", "splits", "threshold_policies", "processing", "data_sha256", "code_sha256", "limitations")})
    print(f"Saved test={len(test_export)} rows, candidates={len(current)}, selected={evidence['current']['selected']}, "
          f"risk={evidence['current']['risk_flagged']}; {evidence['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
