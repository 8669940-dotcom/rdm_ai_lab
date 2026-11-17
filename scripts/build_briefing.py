"""Build the RDM report, five-minute app briefing and formula/source appendix.

Read-only for data and trained models. Independently recompute the published
metrics before writing documents; reconstruct audit models in memory only.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORT_DATE = "2026-11-17"  # Client date; reporting month remains 2026-01.
SOURCES = {
    "S01": ("RandomForestClassifier：확률 평균·분기 설정", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"),
    "S02": ("Decision Trees：Gini 불순도와 분기 목적", "https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation"),
    "S03": ("average_precision_score：AP 정의", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html"),
    "S04": ("auc：사다리꼴 면적", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html"),
    "S05": ("roc_auc_score：ROC 면적", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html"),
    "S06": ("brier_score_loss：이진 확률 제곱오차", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html"),
    "S07": ("Permutation importance：반복 셔플 중요도", "https://scikit-learn.org/stable/modules/permutation_importance.html"),
    "S08": ("NumPy quantile：선형 분위수", "https://numpy.org/doc/stable/reference/generated/numpy.quantile.html"),
    "S09": ("pandas Series.quantile：선형 사분위수 API", "https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html"),
    "S10": ("SimpleImputer：학습 중앙값·결측 표시", "https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html"),
    "S11": ("StandardScaler：평균·표준편차 변환", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"),
    "S12": ("TimeSeriesSplit：시간 순서와 gap 개념", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html"),
    "S13": ("LogisticRegression：이진 확률과 규제 설정", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"),
    "S14": ("HistGradientBoostingClassifier：손실·학습률·트리 설정", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html"),
    "S15": ("LightGBM LGBMClassifier：비교 모델 API", "https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html"),
    "S16": ("LightGBM Parameters：이진 목적함수·학습 설정", "https://lightgbm.readthedocs.io/en/stable/Parameters.html"),
    "S17": ("calibration_curve：분위수 구간별 평균 확률·관측 비율", "https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html"),
    "S18": ("OneHotEncoder：범주 지시열·미관측 범주 처리", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html"),
    "S19": ("NIST：사분위수·IQR 울타리와 이상치 해석", "https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm"),
    "S20": ("confusion_matrix：TP·FP·FN·TN 정의", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, value):
    (ROOT / path).write_text(value, encoding="utf-8", newline="\n")


def write_json(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_report(path, text):
    """Keep mathematical/code blocks literal; space Korean prose and numeric labels."""
    output, literal = [], False
    for line in text.splitlines():
        if line.startswith("```") or line.strip() == "$$":
            literal = not literal
            output.append(line)
            continue
        if not literal:
            line = re.sub(r"(?<=[가-힣])(?=[A-Za-z0-9])", " ", line)
            line = re.sub(r",(?=[^\s0-9])", ", ", line)
            line = re.sub(r"\b(AP|ROC-AUC|LR|RF|LightGBM)(?=\d)", r"\1 ", line)
        output.append(line)
    write(path, "\n".join(output) + "\n")


def table(columns, rows):
    esc = lambda value: str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(esc, columns)) + " |",
                      "| " + " | ".join("---" for _ in columns) + " |",
                      *["| " + " | ".join(map(esc, row)) + " |" for row in rows]])


def code_link(path, symbol):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    found = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol]
    if not found:
        raise ValueError(f"Missing source symbol: {path}:{symbol}")
    return f"[`{path}:{symbol}`](../{path}#L{found[0].lineno})"


def source_link(key):
    title, url = SOURCES[key]
    return f"[{key} {title}]({url})"


def manual_metrics(labels, probabilities):
    """Tie-aware AP, trapezoidal PR area, pairwise ROC-AUC and binary Brier.

    No sklearn metric calls: the group precision at each distinct score weights
    that group's recall increase. A positive/negative tied pair contributes .5.
    """
    if not labels or len(labels) != len(probabilities) or any(y not in (0, 1) for y in labels):
        raise ValueError("Require equally sized nonempty binary labels and scores")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Require finite probabilities between zero and one")
    positives = sum(labels)
    groups = defaultdict(list)
    for label, probability in zip(labels, probabilities):
        groups[probability].append(label)
    tp = count = 0
    ap = area = previous_recall = 0.
    previous_precision = 1.
    if positives:
        for probability in sorted(groups, reverse=True):
            group = groups[probability]
            tp += sum(group)
            count += len(group)
            recall, precision = tp / positives, tp / count
            ap += (recall - previous_recall) * precision
            area += (recall - previous_recall) * (previous_precision + precision) / 2
            previous_recall, previous_precision = recall, precision
    positive_scores = [p for y, p in zip(labels, probabilities) if y]
    negative_scores = [p for y, p in zip(labels, probabilities) if not y]
    roc = (math.fsum(1. if p > q else .5 if p == q else 0. for p in positive_scores for q in negative_scores)
           / (len(positive_scores) * len(negative_scores))) if positive_scores and negative_scores else None
    return {"rows": len(labels), "positives": positives, "prevalence": positives / len(labels),
            "ap": ap if positives else None, "pr_auc_trapezoid": area if positives else None,
            "roc_auc": roc, "brier": math.fsum((p-y)**2 for y, p in zip(labels, probabilities)) / len(labels)}


def assert_close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Evidence mismatch: {label}: {actual} != {expected}")


def audit_cleaning(raw, result):
    """Rebuild cost rules and numeric medians directly, without the transformer."""
    import numpy as np
    import pandas as pd
    rows = read_csv("reports/cleaning_audit.csv")
    medians = {}
    for role, saved in result["processing"]["fitted_by_role"].items():
        cutoff = pd.Timestamp(result["splits"][role + "_train"]["prediction_start"])
        part = raw[raw["_source"].eq("train") & (raw["maintenance_date"] + pd.Timedelta(days=30) < cutoff)].copy()
        q1, q3 = np.quantile(part["maintenance_cost"].dropna(), [.25, .75])
        assert_close(q1, saved["q1"], "training Q1")
        assert_close(q3, saved["q3"], "training Q3")
        low, high = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        cost = part["maintenance_cost"].astype(float).to_numpy()
        cleaned = [min(q3, max(q1, x/100 if x > 100*q3 else x)) if x < low or x > high else x for x in cost]
        medians[role] = {"maintenance_cost": float(np.nanmedian(cleaned))}
        for field in ("downtime_hours", "repair_duration_hours", "days_since_last_maintenance"):
            series = part[field].replace(999., np.nan) if field != "days_since_last_maintenance" else part[field]
            medians[role][field] = float(series.median())
    for item in rows:
        role, field = item["role"].split("_")[0], item["field"]
        saved = result["processing"]["fitted_by_role"][role]
        match = raw[raw["_source"].eq(item["source"]) & raw["_source_row"].eq(int(item["source_row"]))]
        assert len(match) == 1
        original = match.iloc[0][field]
        if pd.isna(original) or (field in ("downtime_hours", "repair_duration_hours") and original == 999):
            assert item["cleaned"] == "N/A"
            expected = medians[role][field]
        else:
            assert_close(float(item["original"]), original, "audit original")
            assert original < saved["lower_fence"] or original > saved["upper_fence"]
            corrected = original/100 if original > saved["hundredfold_above"] else original
            expected = min(saved["q3"], max(saved["q1"], corrected))
            assert_close(float(item["cleaned"]), expected, "cost replacement")
        assert_close(float(item["model_imputed"]), expected, "model numeric input")
    expected_rows = sum(x["cost_outliers"] + sum(x["time_999_to_na"].values()) + sum(x["numeric_blank"].values())
                        for x in result["processing"]["applications"].values())
    assert len(rows) == expected_rows
    return len(rows)


def audit(result):
    import joblib
    import numpy as np
    from sklearn.inspection import permutation_importance
    from threadpoolctl import threadpool_limits
    from rdm_model import TARGET, build_features, cost_sweep, load_data
    from train import fit_for_cutoff

    input_paths = ["README.txt", "analysis_config.json", "rdm_model.py", "scripts/train.py",
                   "reports/analysis_results.json", "reports/validation_predictions_2024H2.csv",
                   "reports/threshold_costs.csv", "reports/data_profile.json", "reports/monthly_validation.csv", "reports/eda_summary.csv", "reports/cleaning_audit.csv",
                   "artifacts/rdm_model.joblib", "artifacts/submission_model.joblib",
                   "data/drone_train.csv", "data/drone_test.csv", "data/drone_this_month.csv"]
    before = {p: digest(ROOT / p) for p in input_paths}
    rows = read_csv("reports/validation_predictions_2024H2.csv")
    labels = [int(float(row[TARGET])) for row in rows]
    probabilities = [float(row["prob"]) for row in rows]
    recalculated = manual_metrics(labels, probabilities)
    for metric in ("ap", "pr_auc_trapezoid", "roc_auc", "brier"):
        assert_close(recalculated[metric], result["evaluation"]["metrics"][metric], metric)
    months = defaultdict(list)
    for row in rows:
        months[row["maintenance_date"][:7]].append(row)
    monthly = []
    for original in result["evaluation"]["monthly"]:
        group = months[original["month"]]
        if original["unit"] == "latest_drone":
            latest = {}
            for row in sorted(group, key=lambda r: r["maintenance_date"]):
                latest[row["equipment_id"]] = row
            group = list(latest.values())
        ordered = sorted(group, key=lambda r: (r["equipment_id"], r["maintenance_date"]))
        k = min(result["config"]["capacity"], len(ordered))
        by_model = sorted(ordered, key=lambda r: (-float(r["prob"]), r["equipment_id"], -int(r["maintenance_date"].replace("-", ""))))[:k]
        by_age = sorted(ordered, key=lambda r: (-float(r["equipment_age_years"]), r["equipment_id"], -int(r["maintenance_date"].replace("-", ""))))[:k]
        y = np.array([int(float(r[TARGET])) for r in ordered])
        rng = np.random.default_rng(result["config"]["random_seed"] + int(original["month"].replace("-", "")) + (original["unit"] == "maintenance_record"))
        draws = [int(y[rng.choice(len(ordered), k, replace=False)].sum()) for _ in range(1000)]
        item = {"month": original["month"], "unit": original["unit"], "candidates": len(ordered), "selected": k,
                "positives": int(y.sum()), "model_hits": sum(int(float(r[TARGET])) for r in by_model),
                "age_hits": sum(int(float(r[TARGET])) for r in by_age),
                "random_expected_hits": k * int(y.sum()) / len(ordered), "random_hits_mean": float(np.mean(draws))}
        for key in ("candidates", "selected", "positives", "model_hits", "age_hits", "random_expected_hits", "random_hits_mean"):
            assert_close(item[key], original[key], f"{original['month']}/{original['unit']}/{key}")
        monthly.append(item)
    costs = read_csv("reports/threshold_costs.csv")
    for row in costs:
        assert_close(int(row["false_negatives"]) * int(row["miss_cost"]) + int(row["flagged"]) * 60, float(row["cost"]), "cost row")
    raw = load_data()
    X = build_features(raw)
    cleaning_rows = audit_cleaning(raw, result)
    train_eda = raw[raw["_source"].eq("train")].copy()
    train_eda["month"] = train_eda["maintenance_date"].dt.strftime("%m")
    train_eda["fault_category"] = train_eda["fault_category"].fillna("N/A")
    eda_rows = read_csv("reports/eda_summary.csv")
    for item in eda_rows:
        field = {"드론 유형": "equipment_type", "정비 월": "month", "고장 유형": "fault_category"}[item["기준"]]
        subset = train_eda[train_eda[field].eq(item["구분"])]
        assert len(subset) == int(item["정비 건수"])
        assert int(subset[TARGET].sum()) == int(item["재고장 건수"])
        assert_close(float(subset[TARGET].mean()), float(item["재고장률"]), "EDA subgroup rate")
    bounds = np.quantile(probabilities, np.linspace(0., 1., 6))
    bin_ids = np.searchsorted(bounds[1:-1], probabilities)
    calibration = []
    for bin_id in range(5):
        indices = [i for i, b in enumerate(bin_ids) if b == bin_id]
        if indices:
            calibration.append({"rows": len(indices), "mean_predicted": math.fsum(probabilities[i] for i in indices)/len(indices),
                                "observed_rate": sum(labels[i] for i in indices)/len(indices)})
    assert len(calibration) == len(result["evaluation"]["calibration"]), "Calibration bin count changed"
    for rebuilt, saved in zip(calibration, result["evaluation"]["calibration"]):
        assert_close(rebuilt["mean_predicted"], saved["mean_predicted"], "calibration mean")
        assert_close(rebuilt["observed_rate"], saved["observed_rate"], "calibration observed")
    config, spec = result["config"], result["selected_spec"]
    selection_mask = raw["_source"].eq("train") & raw["maintenance_date"].between(config["selection_start"], config["selection_end"])
    policies = {}
    with threadpool_limits(limits=2):
        development_model, _ = fit_for_cutoff(raw, X, config["selection_start"], spec, config)
        p = development_model.predict_proba(X.loc[selection_mask])[:, 1]
        development_metrics = manual_metrics(raw.loc[selection_mask, TARGET].astype(int).tolist(), p.tolist())
        assert_close(development_metrics["ap"], result["development"]["metrics"]["ap"], "development AP")
        for cost in config["miss_costs"]:
            best, _ = cost_sweep(raw.loc[selection_mask, TARGET].astype(int), p, cost, config["inspection_cost"])
            saved = result["threshold_policies"][str(cost)]
            for key in ("threshold", "cost", "flagged", "true_positives", "false_negatives"):
                assert_close(best[key], saved[key], f"policy {cost}/{key}")
            policies[str(cost)] = best
        evaluation_mask = raw["_source"].eq("train") & raw["maintenance_date"].ge(config["validation_start"])
        evaluation_model, _ = fit_for_cutoff(raw, X, config["validation_start"], spec, config)
        importance = permutation_importance(evaluation_model, X.loc[evaluation_mask], raw.loc[evaluation_mask, TARGET].astype(int),
                                            scoring="average_precision", n_repeats=5, random_state=config["random_seed"], n_jobs=1)
        for item in result["permutation_importance"]:
            index = list(X.columns).index(item["feature"])
            assert_close(importance.importances_mean[index], item["ap_decrease_mean"], "permutation importance")
        bundle = joblib.load(ROOT / "artifacts/rdm_model.joblib")
        model = bundle["model"]
        month = raw[raw["_source"].eq("month")]
        current = result["current"]["equipment"][0]
        target_index = month.index[month["_source_row"].eq(current["_source_row"])][0]
        x = X.loc[[target_index]]
        probability = float(model.predict_proba(x)[0, 1])
        assert_close(probability, current["prob"], "example probability")
        factor_audit = []
        for factor in current["factors"]:
            replaced = x.copy()
            replaced[factor["feature"]] = bundle["reference"][factor["feature"]]
            reference_probability = float(model.predict_proba(replaced)[0, 1])
            assert_close(probability - reference_probability, factor["delta_probability"], "local sensitivity")
            factor_audit.append({**factor, "original_probability": probability, "replaced_probability": reference_probability})
        parameters = {key: model.named_steps["model"].get_params()[key] for key in
                      ("n_estimators", "criterion", "max_depth", "min_samples_leaf", "max_features", "bootstrap", "class_weight", "random_state")}
    ranking = sorted(result["current"]["equipment"], key=lambda r: (-r["prob"], r["equipment_id"]))
    assert len({r["equipment_id"] for r in ranking}) == len(ranking) == 81
    assert [r["equipment_id"] for r in ranking[:20]] == [r["equipment_id"] for r in ranking if r["selected"]]
    ranked_threshold = result["threshold_policies"]["300"]["threshold"]
    assert sum(r["prob"] >= ranked_threshold for r in ranking) == result["current"]["risk_flagged"]
    chosen = sorted(result["comparisons"], key=lambda r: (-r["ap"], -r["latest_top20_hits"], r["brier"], r["key"]))[0]["key"]
    assert chosen == result["selected_model"]
    after = {p: digest(ROOT / p) for p in input_paths}
    assert before == after, "Audit changed source data or analysis artifacts"
    return {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "tolerance_absolute": 1e-12, "input_sha256": before,
            "manual_holdout_metrics": recalculated, "development_reconstruction": development_metrics,
            "reconstructed_policies": policies, "monthly_recomputed": monthly,
            "cost_rows_recomputed": len(costs), "permutation_features_reproduced": len(result["permutation_importance"]),
            "eda_groups_recomputed": len(eda_rows), "calibration_bins_recomputed": calibration,
            "cleaning_rows_recomputed": cleaning_rows,
            "example_drone": current["equipment_id"], "example_factor_recomputed": factor_audit,
            "random_forest_parameters": parameters, "data_and_models_unchanged": True}


def main():
    result = read_json("reports/analysis_results.json")
    profile = read_json("reports/data_profile.json")
    evidence = audit(result)
    evidence["generator_sha256"] = digest(Path(__file__))
    write_json("reports/briefing_evidence.json", evidence)
    write_json("reports/method_sources.json", {
        "reviewed_date": REPORT_DATE, "implementation_version": result["environment"],
        "project_sources": {"D01": "README.txt: task, target, costs and capacity", "D02": "analysis_results.json: executed model/metrics/policy",
                            "D03": "data_profile.json: observed quality counts", "D04": "source functions linked in formula appendix"},
        "external_sources": {key: {"title": title, "url": url} for key, (title, url) in SOURCES.items()},
        "scope": "Official algorithm/API definitions and IQR fences. Costs, 999 missingness, hundredfold errors, quartile replacement, history windows and fixed model settings are task/user assumptions or project choices, not military standards."})
    build_documents(result, profile, evidence)
    paths = ["reports/2026-01_분석결과보고.md", "reports/2026-01_지휘관발표스크립트.md", "reports/계산식_출처_대조표.md",
             "reports/briefing_evidence.json", "reports/method_sources.json", "scripts/build_briefing.py", "scripts/briefing_content.py",
             "app.py", "app_styled.py", "rdm_ui/__init__.py", "rdm_ui/dashboard.py", "rdm_ui/readability.css"]
    write_json("reports/briefing_manifest.json", {p: digest(ROOT / p) for p in paths})
    print("Built report, five-minute script and formula/source appendix; independent calculation audit passed.")


def build_documents(result, profile, evidence):
    from briefing_content import documents
    for path, body in documents(result, profile, evidence, table, source_link, code_link).items():
        write_report(path, body)


if __name__ == "__main__":
    main()
