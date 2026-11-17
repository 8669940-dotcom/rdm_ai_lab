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
    "S09": ("pandas Series.quantile：운영 상한 계산 API", "https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html"),
    "S10": ("SimpleImputer：학습 중앙값·결측 표시", "https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html"),
    "S11": ("StandardScaler：평균·표준편차 변환", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"),
    "S12": ("TimeSeriesSplit：시간 순서와 gap 개념", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html"),
    "S13": ("LogisticRegression：이진 확률과 규제 설정", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"),
    "S14": ("HistGradientBoostingClassifier：손실·학습률·트리 설정", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html"),
    "S15": ("LightGBM LGBMClassifier：비교 모델 API", "https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html"),
    "S16": ("LightGBM Parameters：이진 목적함수·학습 설정", "https://lightgbm.readthedocs.io/en/stable/Parameters.html"),
    "S17": ("calibration_curve：분위수 구간별 평균 확률·관측 비율", "https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html"),
    "S18": ("OneHotEncoder：범주 지시열·미관측 범주 처리", "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html"),
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


def audit(result):
    import joblib
    import numpy as np
    from sklearn.inspection import permutation_importance
    from threadpoolctl import threadpool_limits
    from rdm_model import TARGET, build_features, cost_sweep, load_data
    from train import fit_for_cutoff

    input_paths = ["README.txt", "analysis_config.json", "rdm_model.py", "scripts/train.py",
                   "reports/analysis_results.json", "reports/validation_predictions_2024H2.csv",
                   "reports/threshold_costs.csv", "reports/data_profile.json", "reports/monthly_validation.csv", "reports/eda_summary.csv",
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
    train_eda = raw[raw["_source"].eq("train")].copy()
    train_eda["month"] = train_eda["maintenance_date"].dt.strftime("%m")
    train_eda["fault_category"] = train_eda["fault_category"].fillna("해당 없음")
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
        "scope": "Official algorithm/API definitions. Costs, q99, history windows and model configuration are educational assumptions or project choices, not military standards."})
    build_documents(result, profile, evidence)
    paths = ["reports/2026-01_분석결과보고.md", "reports/2026-01_지휘관발표스크립트.md", "reports/계산식_출처_대조표.md",
             "reports/briefing_evidence.json", "reports/method_sources.json", "scripts/build_briefing.py"]
    write_json("reports/briefing_manifest.json", {p: digest(ROOT / p) for p in paths})
    print("Built report, five-minute script and formula/source appendix; independent calculation audit passed.")


def build_documents(result, profile, evidence):
    build_formula_appendix(result, evidence)
    build_analysis_report(result, profile, evidence)
    build_speech(result, evidence)


def build_formula_appendix(result, evidence):
    formulas = [
        ("F01", "건수·중복·관측 재고장률", r"D_{exact}=n-|\{row_i\}|,\quad D_{key}=n-|\{(e_i,t_i)\}|,\quad \hat\pi=\frac{\sum_i y_i}{n}",
         "완전 중복은 모든 원본 열, 키 중복은 드론·정비일 기준이다. 드론 ID 반복은 이력이다. 학습 양성률은 525/2346이며 시험에는 y가 없어 계산하지 않는다.",
         code_link("scripts/profile_data.py", "profile_file"), "data_profile.json: files.*.exact_duplicate_extras / equipment_date_duplicate_extras / target_positive_rate", "제공 CSV를 집계한 프로젝트 정의. 외부의 부대 실적 수치가 아니다."),
        ("F02", "선형 분위수·IQR 후보·q99 상한", r"h=(n-1)q,\ j=\lfloor h\rfloor,\ a=h-j,\ Q_q=(1-a)x_{(j)}+ax_{(\min(j+1,n-1))}\quad(0\text{부터 인덱스})",
         r"정렬한 유효 수치에서 선형 보간한다. $IQR=Q_{.75}-Q_{.25}$, 후보는 $x<Q_{.25}-1.5IQR$ 또는 $x>Q_{.75}+1.5IQR$다. 실제 상한 처리는 $x'=\min(x,Q_{.99}^{train})$로 운용중단·비용 두 열만 적용한다. 등급/이진 IQR은 제외한다. 1.5와 .99는 프로젝트의 탐색·비교 설정이다. 999를 자동 결측, 19000을 100으로 나누는 규칙은 이 프로젝트에 없다.",
         code_link("scripts/profile_data.py", "quantile") + ", " + code_link("rdm_model.py", "QuantileCapper"),
         "data_profile.json: files.*.numeric; analysis_results.json: processing.operation_train_q99_caps",
         source_link("S08") + "; " + source_link("S09") + ". 선형 분위수 구현의 출처이며 q99가 군 표준이라는 근거는 아니다."),
        ("F03", "결측 대체·표시·표준화·범주 인코딩", r"m_j=\operatorname{median}(X_j^{train}),\quad x_j^*=\begin{cases}m_j&x_j\text{ 결측}\\x_j&\text{그 외}\end{cases},\quad b_j=\mathbf1[x_j\text{ 결측}]",
         r"상한 처리 다음에 학습 중앙값 대체·결측 표시를 적용한다. 자동 결측 지시열은 학습 중 결측이 있었던 열에 추가하며 gap_missing·과거 이력 존재 표시는 별도로 만든다. $z_j=(x_j^*-\mu_j^{train})/\sigma_j^{train}$, $\sigma_j=\sqrt{n^{-1}\sum_i(x_{ij}^*-\mu_j)^2}$로 표준화한다. 상수 열의 scale은 1이다. 범주는 $z_{jc}=\mathbf1[x_j=c]$인 원핫 인코딩이며 새로운 범주는 해당 범주 블록을 all-zero로 처리한다. 공통 전처리이며 위험 요인의 수동 가중치가 아니다.",
         code_link("rdm_model.py", "make_model"), "artifacts/model_card.json: processing; 실제 fitted 통계는 rdm_model.joblib의 prep", source_link("S10") + "; " + source_link("S11") + "; " + source_link("S18") + ". 예방정비 공란의 해당 없음 분류는 프로젝트 판단이다."),
        ("F04", "현재일 이전 이력 피처", r"H_e(t)=\{j:e_j=e,\ t_j<t\},\quad N_{90}(e,t)=\sum_{j\in H_e(t)}\mathbf1[t-90d\le t_j<t]",
         r"총 과거 정비 수는 $|H_e(t)|$다. 관측 사후·긴급정비 수는 90일 창의 합에 $\mathbf1[M_j\in\{사후,긴급\}]$를 곱한다. 이전 고장 경과일은 $t-\max\{t_j:j\in H_e(t),M_j\in\{사후,긴급\}\}$다. 이전 예방정비 후 시간은 $\max(0,h_e(t)-h_e(t_{prevPM}))$다. 이전180일 정비 실패 수는 해당 창에서 $\mathbf1[success_j=0]$의 합이다. 기록이 없는 창의 횟수는 0이며, 이전 고장/예방정비가 없으면 해당 경과일/시간은 NaN·존재 표시는 0이다. 기록이 있으면 존재 표시는 1이다. 파일 경계에서 이력을 초기화하지 않으며 당일 기록과 숨겨진 정답은 제외한다.",
         code_link("rdm_model.py", "build_features"), "analysis_results.json: processing.feature_columns; build_features 실행 결과",
         "README.txt 3절의 과거 이력 과제를 구체화한 Codex 설계다. 90/180일은 외부 군 정비 규정에서 인용한 값이 아니다."),
        ("F05", "월 주기 피처", r"s_m=\sin(2\pi m/12),\quad c_m=\cos(2\pi m/12),\quad m\in\{1,\ldots,12\}",
         "12월과 1월의 인접성을 표현하기 위한 달력 주기 변환이다. 군 기종의 계절 고장 법칙을 입증한 것은 아니다.", code_link("rdm_model.py", "build_features"), "processing.feature_columns: month_sin / month_cos", "달력의 12개월 주기를 이용한 프로젝트 피처 설계."),
        ("F06", "정답 성숙을 고려한 학습 경계", r"\mathcal T(T)=\{i:source_i=train,\quad t_i+30d<T\}",
         "2024-07-01 예측에는 2024-05-31까지의 1611행을 사용한다. 6월1일의 관측 종료일은 7월1일이므로 엄격한 < 조건에서 제외한다. 2025 제출은 2223행, 2026 운영은 2346행이다. 실제 구현은 날짜 마스크이며 TimeSeriesSplit 객체를 호출하지 않았다.",
         code_link("scripts/train.py", "training_mask") + ", " + code_link("scripts/train.py", "fit_for_cutoff"), "analysis_results.json: splits", "30일은 README의 정답 관측 길이. 엄격 경계는 프로젝트의 보수적 설계. " + source_link("S12") + "는 시간 순서/gap의 참고이며 gap=30행을 사용한 것은 아니다."),
        ("F07", "Random Forest 확률·분기·가중치", r"\hat p(x)=\frac1{240}\sum_{b=1}^{240}q_b(x),\quad q_b(x)=\frac{\sum_{i\in leaf_b(x)}w_{bi}y_i}{\sum_{i\in leaf_b(x)}w_{bi}}",
         r"$w_{bi}$는 부트스트랩 학습 표본의 중복 반영 가중치다. 트리 간 평균 가중치는 각각 1/240이다. Gini는 $G(t)=1-\sum_k q_{tk}^2$다. 노드 내부에서 $\Delta G=G(t)-(W_L/W_t)G(L)-(W_R/W_t)G(R)$를 비교하며 W는 가중 표본 수다. 최대 깊이6, 최소 잎 표본8, max_features=sqrt, bootstrap=True, class_weight=None, seed42다. 사람이 고장 횟수40%·연식30% 같은 요인 가중치를 부여하지 않았다.",
         code_link("rdm_model.py", "make_model"), "briefing_evidence.json: random_forest_parameters; artifacts/rdm_model.joblib", source_link("S01") + "; " + source_link("S02")),
        ("F08", "임계값별 혼동행렬·정밀도·재현율", r"a_i(\tau)=\mathbf1[p_i\ge\tau],\quad TP=\sum y_i a_i,\quad FP=\sum(1-y_i)a_i,\quad FN=\sum y_i(1-a_i),\quad TN=\sum(1-y_i)(1-a_i)",
         r"정밀도 $P=TP/(TP+FP)$, 재현율 $R=TP/(TP+FN)$, 거짓양성률 $FPR=FP/(FP+TN)$다. 실제 평가 코드의 0분모 정밀도/재현율은 None으로 표시한다. 위험 기준 이상의 수 $A=TP+FP$와 점검 정원 K는 다르다.",
         code_link("rdm_model.py", "evaluate_policy") + ", " + code_link("scripts/rdm_common.py", "threshold_cost"), "analysis_results.json: threshold_policies / evaluation.frozen_policy_costs", source_link("S03") + "; " + source_link("S05") + ". 프로젝트의 >= 비교를 명시했다."),
        ("F09", "주 지표 AP와 사다리꼴 PR-AUC", r"AP=\sum_g(R_g-R_{g-1})P_g,\qquad PR\text{-}AUC_{trap}=\sum_g(R_g-R_{g-1})\frac{P_g+P_{g-1}}2",
         "점수를 높은 순서로 내리며 서로 다른 점수 그룹 전체를 함께 포함하고, 재현율이 증가하는 방향으로 계산한다. 시작점은 R0=0,P0=1이다. AP는 해당 단계의 정밀도로 가중하고 사다리꼴 면적은 인접 정밀도를 평균한다. 같은 점수 두 건 중 양성 한 건이면 AP=.5, 사다리꼴 면적=.75여서 같지 않다. 본 보고의 주 PR 지표는 AP다.",
         code_link("rdm_model.py", "metrics") + ", " + code_link("scripts/build_briefing.py", "manual_metrics"), "analysis_results.json: evaluation.metrics.ap / pr_auc_trapezoid; briefing_evidence.json: manual_holdout_metrics", source_link("S03") + "; " + source_link("S04")),
        ("F10", "ROC-AUC와 동점 처리", r"ROC\text{-}AUC=\frac{\sum_{i:y_i=1}\sum_{j:y_j=0}\left(\mathbf1[p_i>p_j]+\frac12\mathbf1[p_i=p_j]\right)}{n_+n_-}",
         "양성·음성 쌍의 순위 구분력을 나타내며 동점은 0.5로 센다. 본 감사에서는 176×467=82192쌍을 독립 계산해 원 지표와 대조한다. AP/ROC-AUC는 정확도 78.77%/88.14%라는 뜻이 아니다.",
         code_link("rdm_model.py", "metrics") + ", " + code_link("scripts/build_briefing.py", "manual_metrics"), "evaluation.metrics.roc_auc / briefing_evidence.json: manual_holdout_metrics.roc_auc", source_link("S05") + ". 쌍 비교는 본 보고에서 구현한 동등한 이진 순위 검산식이다."),
        ("F11", "Brier 확률 오차", r"Brier=\frac1n\sum_{i=1}^n(p_i-y_i)^2",
         "이진 문제의 0~1 척도이며 낮을수록 제곱 확률 오차가 작다. 이 값만으로 확률 보정이 완료됐다고 판단하지 않는다.", code_link("rdm_model.py", "metrics"), "evaluation.metrics.brier / briefing_evidence.json: manual_holdout_metrics.brier", source_link("S06")),
        ("F12", "월 최신 드론 선정과 적중", r"i^*_{e,m}=\operatorname{argmax}_{i:e_i=e,month(t_i)=m}t_i,\quad S_m=TopK_p\{i^*_{e,m}\},\quad H_m=\sum_{i\in S_m}y_i",
         "확률 동률은 드론 번호 오름차순, 최신 날짜 동률 충돌은 임의로 결정하지 않는다. K=min(20,후보 수). 6개월 합은 ΣHm이며 같은 드론의 다른 월 선정은 별개 사건이다. 정비 기록 상위20건은 중복 드론을 허용하는 별도 집계다. 이번 달은 최신81대에서20대를 고르고 y가 없어 실제 적중을 산출하지 않는다.",
         code_link("scripts/rdm_common.py", "rank_latest") + ", " + code_link("rdm_model.py", "monthly_topk"), "monthly_validation.csv: latest_drone / maintenance_record; current.equipment", "README의 20대 조건을 고유 드론 운영/정비 기록 채점으로 구분한 프로젝트 정의."),
        ("F13", "무작위 기준선 기대값과 반복 평균", r"E[H_m^{random}]=K_m\frac{M_m}{N_m},\quad \bar H_m=\frac1{1000}\sum_{r=1}^{1000}H_{m,r}",
         "Nm은 같은 후보 수, Mm은 그 후보의 양성 수다. 중복 없이 Km개를 뽑아 seed=42+YYYYMM(+정비기록 단위이면1)로 반복한다. 1000회 평균은 이론 기대값과 조금 다를 수 있다. 원 파일의 p025/p975는 무작위 선택 결과 분포의 분위수이며 평균 추정의 신뢰구간이 아니다. 연수 기준선은 같은 후보의 경과연수 내림차순이다.",
         code_link("rdm_model.py", "monthly_topk"), "evaluation.monthly: random_expected_hits / random_hits_mean / random_repeats", "README가 요구한 기준선. 기대값은 각 후보의 선정확률 K/N을 합한 프로젝트 수학적 유도다. 1000회·시드는 재현 설정이다."),
        ("F14", "위험 임계값 비용과 정원 운영 비용", r"C_c(\tau)=c\,FN(\tau)+60\,A(\tau),\quad \tau_c^*=\operatorname{argmin}_{\tau}C_c(\tau),\quad c\in\{300,600\}\ (만원)",
         r"앞 검증511건의 판정 변경점과 모두 표시/표시 없음 정책을 탐색한다. 동률은 표시 수가 적고 임계값이 큰 정책 순이다. 뒤 검증에는 고정한다. 정원 운영 비용은 $C_{K,c}=c(M_m-H_m)+60K_m$로 같은 최신 후보에 별도 계산한다. 두 집계는 분모가 다르므로 합산하지 않는다. 완전 보정과 완전 예방을 가정하면 기대 손실 비교로 p≥60/c(0.20/0.10)를 유도할 수 있지만, 실제 사용값은 이 값이 아니라 검증 최소비용 기준이다. 미탐지 손실·점검비·완전 예방은 교육용 가정이며 실제 절감액이 아니다.",
         code_link("rdm_model.py", "cost_sweep") + ", " + code_link("rdm_model.py", "monthly_topk"), "threshold_costs.csv / threshold_policies / evaluation.frozen_policy_costs", "비용300·600·60만원은 README.txt 1·3절. 경험 최소화·동률 규칙은 프로젝트 구현이며 군 원가 규정이 아니다."),
        ("F15", "개별 민감도와 전체 순열 중요도", r"\Delta_j(x)=\hat p(x)-\hat p(x_{j\leftarrow r_j}),\quad \Delta_j^{pp}=100\Delta_j,\quad I_j=s_0-\frac15\sum_{r=1}^5s_{j,r}",
         r"개별 Δ는 2026 운영 모델에서 한 특성을 학습 중앙값/최빈값 rj로 치환한 확률 차이다. 양의 차이 상위3개를 표시한다. 전체 Ij는 하반기 평가 모델에서 특성만 섞어 AP가 줄어드는 정도를 5회 평균한 값이다. 표의 반복 표준편차는 $s_j=\sqrt{5^{-1}\sum_r[(s_0-s_{j,r})-I_j]^2}$이며 신뢰구간이 아니다. 서로 다른 모델·표본·척도이며 합산 가중치로 섞지 않는다. Δ는 SHAP·인과·실제 정비 개입 효과가 아니고, 상관 특성을 하나씩 바꾸면 비현실적 조합 또는 중복 효과가 생길 수 있다.",
         code_link("scripts/train.py", "explain_reference") + ", " + code_link("scripts/train.py", "main"), "current.equipment[].factors / permutation_importance; briefing_evidence.json: example_factor_recomputed", "개별 Δ는 Codex의 프로젝트 설명 방식이다. 전체 순열 중요도는 " + source_link("S07")),
        ("F16", "비교 모델의 출력 형식", r"p_{LR}(x)=\sigma(\beta_0+\beta^Tz),\quad p_{boost}(x)=\sigma\left(F_0(x)+\eta\sum_{m=1}^{M}f_m(x)\right),\quad \sigma(a)=\frac1{1+e^{-a}}",
         "LR은 C=.5, max_iter=2000. HistGB는 이진 log_loss·학습률.04·최대160회·잎15·L2=2. LightGBM은 이진 목적·학습률.035·트리180·잎15·최소자식20·L2=2다. 고정 설정의 8개 모델/피처 구성을 비교한 것이며 전 모델 최적 튜닝이나 각각의 제거 효과를 검증한 것은 아니다. 해당 모델들의 학습은 공식 라이브러리 API를 호출했고 수식의 계수/나무는 데이터로 학습된다.",
         code_link("rdm_model.py", "make_model"), "analysis_results.json: comparisons / selected_spec", source_link("S13") + "; " + source_link("S14") + "; " + source_link("S15") + "; " + source_link("S16")),
        ("F17", "보정곡선용 구간 집계", r"\bar p_b=\frac1{n_b}\sum_{i\in B_b}p_i,\qquad \bar y_b=\frac1{n_b}\sum_{i\in B_b}y_i",
         "하반기 예측 확률을 분위수 기준5개 구간으로 나눠 평균 확률과 관측 재고장률을 비교한다. 동점·빈 구간에 따라 구간 수와 크기가 달라질 수 있다. 이 점검 집계를 계산했을 뿐 확률 보정기를 따로 학습한 것은 아니다. 별도 재검산에서도 저장된 구간 평균과 일치했다.",
         code_link("scripts/train.py", "main"), "analysis_results.json: evaluation.calibration; briefing_evidence.json: calibration_bins_recomputed", source_link("S17") + ". n_bins=5, strategy=quantile은 프로젝트의 점검 설정이다."),
    ]
    header = f"""# 드론 예측정비 계산식·출처·구현 대조표

작성일 {REPORT_DATE} / 분석 기준월 2026-01 / 분석 기준 커밋 af37195.

이 문서는 [분석결과보고](2026-01_분석결과보고.md)와 [5분 발표 대본](2026-01_지휘관발표스크립트.md)의 상세 근거다. 수식은 실제 코드의 계산을 표기했다. 외부 문서는 알고리즘/API의 정의를 뒷받침하고, 비용·정원·기간·피처 창은 제공 과제와 프로젝트 판단에서 왔다.

## 기호와 정답의 출처

e는 드론, t는 정비일, y는 제공된 30일 이내 재고장 정답(0/1), p는 모델 추정 확률, K는 점검 정원이다. y는 학습 CSV에서 읽었다. 시험/이번 달의 숨겨진 정답을 향후 기록으로 재구성하지 않았다. 조건이 성립하면 1인 표시함수를 $\\mathbf1[\\cdot]$로 표기한다.

## 계산식 찾아보기

"""
    text = header + table(["ID", "계산", "코드"], [[key, title, code] for key, title, _, _, code, _, _ in formulas]) + "\n\n"
    for key, title, equation, meaning, code, fields, source in formulas:
        text += f"## {key}. {title}\n\n$$\n{equation}\n$$\n\n{meaning}\n\n- 실제 구현: {code}\n- 실행 결과: `{fields}`\n- 출처와 적용 구분: {source}\n\n"
    metrics = evidence["manual_holdout_metrics"]
    text += "## 독립 재검산과 실제 대입\n\n" + table(["계산", "대입/재계산", "원 실행과 대조"], [
        ["하반기 AP", f"{metrics['ap']:.12f}", "절대오차 1e-12 이내"],
        ["하반기 PR 사다리꼴", f"{metrics['pr_auc_trapezoid']:.12f}", "동점 그룹으로 직접 적분"],
        ["하반기 ROC-AUC", f"{metrics['roc_auc']:.12f}", "82192개 양성·음성 쌍 직접 비교"],
        ["하반기 Brier", f"{metrics['brier']:.12f}", "643개 제곱오차 평균"],
        ["앞 검증 300만원", "40×300 + 111×60 = 18,660만원", "정책 재구성·비용표 전 행 산술 검산"],
        ["앞 검증 600만원", "27×600 + 182×60 = 27,120만원", "정책 재구성·비용표 전 행 산술 검산"],
        ["뒤 검증 300만원", "52×300 + 175×60 = 26,100만원", "앞 기준 고정 적용"],
        ["뒤 검증 600만원", "28×600 + 255×60 = 32,100만원", "앞 기준 고정 적용"],
        ["7월 무작위 기대 적중", f"20×9/77 = {20*9/77:.6f}건", "반복 평균2.357건과 정의 구분"],
    ])
    text += "\n\nAP·ROC·Brier는 sklearn 지표 함수를 호출하지 않는 별도 수식으로 검산했다. 앞/뒤 평가용 모델은 같은 설정으로 메모리에서만 재구성했고, 저장된 운영·제출 모델과 원본을 바꾸지 않았다. 전체 수치·입력 해시는 `briefing_evidence.json`, 문서 해시는 `briefing_manifest.json`이다.\n\n"
    text += "## 공식 출처 목록\n\n확인일 2026-11-17. 당시 설치 구현은 scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.5, LightGBM 4.7.0이다.\n\n"
    text += "\n".join("- " + source_link(key) for key in SOURCES) + "\n\n외부 출처가 지지하지 않는 프로젝트 값: 비용·20대·30일(README), q99·IQR탐색·90/180일·분할일·시드·반복수·고정 모델 설정(Codex 설계). 이를 군 표준 또는 실제 부대 검증값으로 표현하지 않는다.\n"
    write_report("reports/계산식_출처_대조표.md", text)


def build_analysis_report(result, profile, evidence):
    metric = result["evaluation"]["metrics"]
    current = result["current"]
    latest = [r for r in evidence["monthly_recomputed"] if r["unit"] == "latest_drone"]
    records = [r for r in evidence["monthly_recomputed"] if r["unit"] == "maintenance_record"]
    total = lambda items, field: sum(item[field] for item in items)
    example = current["equipment"][0]
    policies = result["threshold_policies"]
    data_table = table(["구분", "기간", "정비 기록", "고유 드론", "열", "완전 중복", "드론·날짜 중복"], [
        [name, f"{p['date_min']} – {p['date_max']}", p["rows"], p["unique_equipment"], p["column_count"],
         p["exact_duplicate_extras"], p["equipment_date_duplicate_extras"]] for name, p in profile["files"].items()])
    comparison = table(["모델·피처 구성", "앞 검증 AP", "앞 검증 ROC-AUC", "앞 검증 Brier", "앞 검증 월20대 적중"], [
        [row["key"], f"{row['ap']:.4f}", f"{row['roc_auc']:.4f}", f"{row['brier']:.4f}", row["latest_top20_hits"]]
        for row in sorted(result["comparisons"], key=lambda r: -r["ap"])])
    splits = table(["모델 역할", "학습 건", "학습 마지막 날", "마지막 정답 관측 종료", "예측 시작"], [
        [name, split["rows"], split["end"], split["latest_label_end"], split["prediction_start"]]
        for name, split in result["splits"].items()])
    monthly = table(["월", "최신 후보", "후보 재고장", "모델 적중", "연수 기준 적중", "무작위1000회 평균"], [
        [r["month"], r["candidates"], r["positives"], r["model_hits"], r["age_hits"], f"{r['random_hits_mean']:.3f}"] for r in latest])
    cost_table = table(["미탐지 가정(만원)", "앞 검증 최적 기준", "TP", "FP", "FN", "TN", "표시 건", "비용(만원)"], [
        [key, f"{row['threshold']:.8f}", row["true_positives"], row["flagged"]-row["true_positives"], row["false_negatives"],
         result["development"]["metrics"]["rows"]-row["flagged"]-row["false_negatives"], row["flagged"], f"{row['cost']:,}"]
        for key, row in policies.items()])
    holdout_cost = table(["가정(만원)", "앞 기준 고정 적용 TP", "FP", "FN", "TN", "표시 건", "뒤 비용(만원)"], [
        [key, row["true_positives"], row["flagged"]-row["true_positives"], row["false_negatives"],
         metric["rows"]-row["flagged"]-row["false_negatives"], row["flagged"], f"{row['cost']:,}"]
        for key, row in result["evaluation"]["frozen_policy_costs"].items()])
    factors = table(["관측 특성", "관측값", "학습 참조값", "원 확률", "치환 후 확률", "차이(%p)"], [
        [r["label"], r["value"], r["reference"], f"{r['original_probability']:.2%}", f"{r['replaced_probability']:.2%}",
         f"{r['delta_probability']*100:.3f}"] for r in evidence["example_factor_recomputed"]])
    importance = table(["특성", "하반기 AP 감소 평균", "반복 표준편차"], [
        [r["label"], f"{r['ap_decrease_mean']:.5f}", f"{r['ap_decrease_std']:.5f}"] for r in result["permutation_importance"][:6]])
    context_lr = next(r for r in result["comparisons"] if r["key"] == "lr_context_cap")
    history_lr = next(r for r in result["comparisons"] if r["key"] == "lr_history_cap")
    waiting = [r for r in current["equipment"] if r["risk_flag"] and not r["selected"]]
    eda = read_csv("reports/eda_summary.csv")
    patterns = []
    for category in ("드론 유형", "정비 월", "고장 유형"):
        group = [r for r in eda if r["기준"] == category]
        highest = max(group, key=lambda r: float(r["재고장률"]))
        lowest = min(group, key=lambda r: float(r["재고장률"]))
        patterns.append([category, highest["구분"], f"{highest['재고장 건수']}/{highest['정비 건수']} ({float(highest['재고장률']):.2%})",
                         lowest["구분"], f"{lowest['재고장 건수']}/{lowest['정비 건수']} ({float(lowest['재고장률']):.2%})"])
    text = f"""# 군 특화 AI 활용 데이터분석 과정 — 드론 예측정비 실습 결과보고

- 보고 대상: 지휘관 및 드론 정비반장
- 작성일: {REPORT_DATE} / 분석 대상월: **2026년 1월**
- 자료: 교육용 가상 드론 데이터. 실제 부대·기종의 운용 성과가 아니다.
- 발표 자료: [5분 지휘관 발표 대본](2026-01_지휘관발표스크립트.md)
- 계산 근거: [계산식·출처 대조표](계산식_출처_대조표.md), [독립 재검산 JSON](briefing_evidence.json), [원 실행 결과](analysis_results.json), [20대 계획](reinspection_plan_2026-01.md)

## 1. 지휘관 보고 요지

정비 직후 확인할 수 있는 상태와 이전 정비 이력으로 30일 이내 재고장 가능성을 예측하여 재점검 순서를 정했다. 제공된 2026년 1월 **106건을 최신 기록 기준81대로 정리하고 상위20대**를 선정했다. 기본 위험 표시 기준 **{policies['300']['threshold']:.2%}** 이상은22대이며, 정원 밖 위험 드론2대는 별도 대기·검토 대상이다. 화면 정렬·필터·요약 상세 조회가 전체 선정 순위를 바꾸지는 않는다.

앞 검증 구간의8개 고정 모델·피처 구성 중 **Random Forest**를 선택했다. 선택에 쓰지 않은2024년 하반기의 정비643건에서 **AP {metric['ap']:.4f}, ROC-AUC {metric['roc_auc']:.4f}**를 확인했다. 같은 기간의 최신 고유 드론 월20대 운영에서는 적중 합 **{total(latest,'model_hits')}건**, 연수 기준 **{total(latest,'age_hits')}건**, 무작위1000회 평균 합 **{total(latest,'random_hits_mean'):.3f}건**이었다.

이는 과거 자료에서의 선정 성능이다. 실제 고장 예방 건수·예산 절감액을 측정하지 않았다. 2025년 시험과2026년1월에는 정답이 없어 실제 성능을 제시하지 않는다. Codex의 판단은 **교육용 우선순위 지원 절차는 재현 가능하나 실제 운용 채택에는 비용·개입 효과·확률 보정·다른 부대 검증이 더 필요하다**는 것이다.

## 2. 자료 구성과 중복 점검

화면: **데이터·과정 → 원본 표**. 산식F01, 구현 {code_link('scripts/profile_data.py','profile_file')}, 결과 `data_profile.json: files`.

{data_table}

드론 ID 반복은 정상 이력이며 중복 오류가 아니다. 이번 달에는 복수 정비 드론18대에서 이전 기록25행이 발생했다. 이력 계산에는 보존하고 선정 시에만 최신 기록을 남겨106건을81대로 바꿨다. 원본 삭제는0행이다. 완전 중복 수와 드론·날짜 중복 수는 각각0으로, 같은 수치라고 정의가 같은 것은 아니다.

### 학습 자료에서 관측한 유형·월·고장별 패턴

{table(['집계','높은 범주','재고장/정비 건수·비율','낮은 범주','재고장/정비 건수·비율'], patterns)}

2023~2024 학습 자료의 범주 내 `재고장 건수/정비 건수`(F01)로 계산했다. 같은 드론의 반복 기록과 연식·정비 구분 등이 섞여 있으므로 유형·계절·고장 종류가 원인이라고 단정하지 않는다. 원본 대조가 완료된22개 범주 집계는 `eda_summary.csv`에 있다. 이 기술 통계는2025년 시험 성능이 아니다.

## 3. 결측·극단값 기준과 이유

화면: **데이터·과정 → 처리 설명**. 산식F02·F03, 구현 {code_link('rdm_model.py','QuantileCapper')}와 {code_link('rdm_model.py','make_model')}.

{table(['항목','실제 관측','처리·이유'], [
    ['고장 코드·유형 공란','학습 각1707건 / 시험 각876건 / 이번 달 각62건, 모두 예방정비','해당 없음 범주. 생성 원인을 확인한 사실로 단정하지 않음'],
    ['직전 정비 경과일','학습150건, 관측 첫 정비','이력 없음 표시+학습 중앙값. 0일과 구분'],
    ['극단 운용중단','학습 최대999시간','오류로 확정하지 않고 q99 상단 제한 대안 비교. 원본 유지'],
    ['극단 정비 비용','학습 최대19000만원','100으로 나누는 근거 없음. q99 제한 대안 비교. 원본 유지'],
    ['기타 수치 결측','학습 구간에서만 대체 통계량 추정','중앙값·결측 표시, 검증/시험 통계량을 학습하지 않음']
])}

운영 학습의 q99 상한은 운용중단 **{result['processing']['operation_train_q99_caps']['downtime_hours']:.3f}시간**, 비용 **{result['processing']['operation_train_q99_caps']['maintenance_cost']:.3f}만원**이다. 이 상한을 넘는 학습 셀은 각각24개·23개다. 이는 셀 수이며 서로 겹친 정비 행을 더해47건 삭제했다고 부르지 않는다. 시험 모델은 연초 학습 구간의 자체 상한을 적용한다.

선형 분위수 계산은 {source_link('S08')}와 {source_link('S09')}, 중앙값/표준화 구현은 {source_link('S10')}·{source_link('S11')}로 확인했다. **q99, IQR 1.5배와 이력 창은 프로젝트 설정**이며 군 정비 표준의 임계치가 아니다. q99 미처리/처리 대조는 로지스틱 구성에서 수행했다. Random Forest 자체의 미처리 대조는 이번8개 구성에 없어, 상한 처리가 RF 성능을 독립적으로 개선했다고 주장하지 않는다.

## 4. 제외 컬럼·과거 이력 피처

화면: **데이터·과정 → 제외 사유·시간 경계**. 평가 행이 아니라 모델 입력에서 뺀 열이다.

{table(['컬럼','제외/사용 이유','근거의 성격'], [
    ['availability_rate','월말 가동률은 정비 직후 가용성 미확인, 이번 달에는 열 자체가 없음','시점·스키마 근거'],
    ['equipment_id','이력 결합·최신 선정·제출 식별에 사용, 모델 ID 암기 입력은 제외','일반화 고려 설계; 새 드론 성능을 입증한 것은 아님'],
    ['unit_code / fault_code','두 열을 묶어 추가한 LR·LightGBM 대안도 비교했으나 최종 구성에서 제외','개별 열의 독립 기여/해로움을 분리 검증하지 않음'],
    ['maintenance_date','정렬·시간 분할·월 sin/cos·과거 이력 계산에 사용, 원 날짜 직접 입력은 제외','피처 설계'],
    ['failure_within_30days','예측할 정답으로만 사용','정답 유출 방지']
])}

부대·코드 포함 LR의 앞 AP는{context_lr['ap']:.4f}, 이력 LR은{history_lr['ap']:.4f}다. 부대·코드가 항상 불필요하거나 누출이라는 뜻은 아니다. 해당 묶음과 고정 모델 설정에서의 비교 결과다.

동일 드론의 현재일보다 앞선 기록만 모아 이전90일 정비/사후·긴급정비 수, 관측 고장 이후 경과일, 이전 예방정비 이후 비행시간 증가, 이전180일 정비 실패 수를 계산했다. 현재일 이후와 당일 다른 기록은 제외하고 파일 경계에서 이력을 초기화하지 않는다. 과거 관측 사후·긴급정비는 숨겨진 미래 정답과 다르다. 자세한 식은F04·F05, 구현 {code_link('rdm_model.py','build_features')}다.

## 5. 시간 분할과30일 정답 성숙

무작위 분할을 사용하지 않았다. 모델·설정·임계값 선택은2024년1월부터5월 **511건**, 추가 평가는2024년7월부터12월 **643건**이다. 학습 자격은 **정비일+30일 < 예측 시작일**이다(F06).

{splits}

2024년6월 결과를 보고7월 예측 모델을 고르지 않았다. 2025년 제출 모델에는 연초에 정답이 아직 성숙하지 않은2024년12월2일 이후 기록을 학습시키지 않았다. 2026년 운영에는 제공된 학습 정답 전체2346행을 썼고2025년 숨겨진 정답을 추가하지 않았다. 시간 순서/gap의 개념은 {source_link('S12')}를 참고했지만 실제 구현은 직접 날짜 마스크다.

## 6. 평가 지표 해설과 계산 출처

화면: **모델 검증 → AP·ROC 곡선·월20대 기준선**. AP/ROC는 정비643건에서, 월20대는 별도 최신 드론 후보에서 계산한다.

{table(['지표','하반기 값','해석·사용 이유','식/공식 출처'], [
    ['AP',f"{metric['ap']:.6f}",'재고장 사례를 높은 순위에 모으는 정밀도·재현율 요약. 주 PR 지표','F09 / '+source_link('S03')],
    ['사다리꼴 PR-AUC',f"{metric['pr_auc_trapezoid']:.6f}",'곡선 점 사이를 선형 연결한 별도 면적. AP와 이름 구분','F09 / '+source_link('S04')],
    ['ROC-AUC',f"{metric['roc_auc']:.6f}",'양성·음성 간 순위 구분력. 불균형에서 운영 적중 수와 함께 해석','F10 / '+source_link('S05')],
    ['Brier',f"{metric['brier']:.6f}",'확률과 정답 제곱오차의 평균, 낮을수록 작음. 보정 완료 증거는 아님','F11 / '+source_link('S06')],
    ['월20대 적중 합',f"{total(latest,'model_hits')}건 / 선정120건",'같은 드론의 다른 월은 별개 선정 사건. 실제 예방 건수 아님','F12 / README·monthly_topk']
])}

하반기643건의 양성은176건, 음성467건이다. AP0.7877은 정확도78.77%가 아니고 ROC-AUC0.8814도 해당 임계값의 정확도가 아니다. 순위·확률 오차·자원 제약을 함께 평가했다.

보정곡선 점검용으로는 확률 분위수5개 구간의 평균 예측 확률과 관측 재고장률을 집계했다(F17). 이는 보정기를 학습하거나 보정 완료를 입증한 절차가 아니다. 저장값은 `evaluation.calibration`, 별도 구간 재검산은 `briefing_evidence.json`에 있다. 정의는 {source_link('S17')}를 참조한다.

## 7. 모델 선정 이유와 가중치의 의미

화면: **모델 검증 → 앞 기간 모델 비교**. 다음 값은 모두 앞 검증 구간의 수치다.

{comparison}

AP 내림차순, 동률이면 월20대 적중·Brier·키 순으로 정했다. RF의 AP0.7158이 비교 LR0.6953·LightGBM0.6513보다 높아 선택됐다. 고정 비교에서는 비선형 상태·이력 관계를 다루는 RF가 유리했지만, 원인을 분리한 실험이나 전 모델 최적 튜닝·유의성 검정을 수행한 것은 아니다. 하반기 점수를 보고 재선정하지 않았고 다른 기간에서도 RF가 항상 우수하다고 결론 내리지 않는다.

**확률은240개 트리의 잎 노드 재고장 비율 평균**이다(F07). 트리 간 평균 가중치는1/240이고 요인마다40%·30%를 사람이 부여한 점수가 아니다. 분기는 Gini 감소로 학습하며 깊이6·최소잎8·sqrt 피처 표본·bootstrap·seed42는 고정 설정이다. 정의는 {source_link('S01')}·{source_link('S02')}, 비교 모델 출력은F16과 공식 API를 참조한다.

## 8. 기준선 대비 성과와 평가 단위

**최신 고유 드론 월20대 운영**에서의 하반기 결과다. 합산 후보는{total(latest,'candidates')}개의 드론-월, 후보 양성은{total(latest,'positives')}건, 선정은120건이다.

{monthly}

{table(['평가 단위','후보 기록/드론-월','모델 적중 합','연수 적중 합','무작위1000회 평균 합'], [
    ['월별 최신 고유 드론',total(latest,'candidates'),total(latest,'model_hits'),total(latest,'age_hits'),f"{total(latest,'random_hits_mean'):.3f}"],
    ['월별 정비 기록 상위20건',total(records,'candidates'),total(records,'model_hits'),total(records,'age_hits'),f"{total(records,'random_hits_mean'):.3f}"]
])}

정비 기록 상위20건은 같은 드론을 반복 포함할 수 있어93건을 고유 드론 적중38건과 같은 지표로 바꿔 부르지 않는다. 강사의 시험 채점 단위는 확인 대상이다. **AP의 분모643건, 최신 후보497드론-월, 선정120건, 이번 달81대는 서로 다르다.**

무작위의 월별 기대 적중은 `선정대수×후보양성/후보수`(F13)다. 예를 들어7월은20×9/77={20*9/77:.4f}건이고 반복 평균은2.357건이다. 6개월 이론 기대값 합{total(latest,'random_expected_hits'):.3f}와 반복 평균 합{total(latest,'random_hits_mean'):.3f}는 정의가 다르다. 반복 결과의 분위수는 평균의 신뢰구간이 아니며 통계적 우월성 검정으로 표현하지 않는다.

## 9. 임계값·비용의 계산과 출처

화면: **모델 검증 → 위험 표시 비용**. 산식F14: `비용=FN×미탐지비용+(TP+FP)×60`(만원). 300·600·60만원은 **README의 교육용 가정**이다. 군 원가 기준이나 실제 부대 손실 추정치가 아니다.

앞 검증511건에서 위험 표시 기준을 고른 결과:

{cost_table}

실제 대입은 **40×300+111×60=18,660만원**, **27×600+182×60=27,120만원**이다. 동률 비용이면 표시 수가 적고 기준이 큰 정책을 택했다. 판정은 `p >= 기준`, 모두 표시/표시 없음 정책도 탐색했다.

앞 기준을 뒤643건에 고정 적용한 결과:

{holdout_cost}

위험 비용은 정원 제한 없는 정비 기록별 평가다. 최신 드론20대 운영 비용은 `(후보양성-선정적중)×300+선정대수×60`으로 같은 후보에서 별도로 계산한다. 6개월 모델 운영 비용 합{sum(r['model_cost_300'] for r in result['evaluation']['monthly'] if r['unit']=='latest_drone'):,.0f}만원은 이 단순 가정의 비교값이며 실제 절감액이 아니다. 비용 시나리오의 단가가 다른 두 총액을 모델 우열로 직접 비교하지 않는다.

완전 보정·완전 예방을 가정한 이론 분기점60/300=.20,60/600=.10은 참고 유도이며 사용한 기준이 아니다. 실제 경험 최적 기준40.14%/22.76%와의 차이는 표본·추정 확률·비용 최적화 조건을 함께 검토할 이유다. 재학습 후 확률 척도와 기준의 재검증이 필요하다.

## 10. 이번 달20대와 위험 요인 설명

현재 정비월의 원본106행에서 최신81대를 고른 뒤 확률순20대를 선정했다. 위험 기준40.14% 이상22대 중 정원 밖2대는 **{', '.join(r['equipment_id'] for r in waiting)}**다. 전체 명단과 권고는 [재점검 계획](reinspection_plan_2026-01.md)에 있다. 초과 위험 드론을 안전하다고 판단한 것이 아니며 대기 순서·운용 전 확인·추가 자원 여부는 정비반장이 검토한다.

시연 드론 **{example['equipment_id']}**, {example['equipment_type']}·{example['unit_code']}, 최근 정비{example['maintenance_date']}·{example['maintenance_type']}, 예측확률 **{example['prob']:.2%}**:

{factors}

산식F15는 `원 예측확률 - 한 특성을 학습 참조값으로 치환한 예측확률`이다. 16.711%p는 고장 위험을 실제로16.711%p 줄일 수 있다는 뜻이 아니다. 여러 차이를 합산하지 않으며 SHAP·인과효과·요인 가중치로 해석하지 않는다. 모터 점검 등 권고는 관측 고장 유형에 연결한 프로젝트 제안으로, 학습한 정비 처방 효과가 아니다.

검증 전체의 특성 중요도는 별도다. 하반기 평가 모델에서 특성을5회 섞어 AP 감소를 계산했다:

{importance}

예를 들어 AP 감소0.01937은 중요도 가중치1.937%가 아니다. 상관·중복 피처는 중요도를 나누거나 가릴 수 있다. 정의는 {source_link('S07')}이며 운영 개별 민감도와 모델·표본·척도가 다르다.

## 11. 앱 시연 순서

{table(['순서','화면·조작','보고할 판단'], [
    [1,'재점검 계획 상단20/81/22/2','선정 정원과 위험 기준의 독립성'],
    [2,'데이터·과정','중복·결측·미래 정보 제외·시간 경계'],
    [3,'모델 검증','앞 비교/뒤 평가, AP·ROC·월20대와 기준선'],
    [4,'재점검 계획: 정렬→유형별 감시드론→돌아가기','전체에서 정한 순위 유지, 감시드론 전체20대'],
    [5,'정비 구분별 사후정비→돌아가기','전체18대 상세, 선정20대 목록 복원'],
    [6,'드론 상세 DR-0115','관측 이력·참조값 치환 민감도·점검 권고'],
    [7,'모델 검증 비용·한계','40.14%/22.76%의 교육용 비용 근거·시험 성능 미확인']
])}

앱은 저장 산출물을 읽고 화면에서 재학습하지 않는다. 필터·정렬은 보기만 바꾼다. 요약에서는 **구분 이름 클릭**으로 해당 범주의 전체 후보를 열고 돌아가기로 복원한다.

## 12. 근거의 성격·검증·운용 전 검토

{table(['구분','내용','출처'], [
    ['직접 관측','행·드론·결측·중복·모델 비교·선정 결과','원 CSV, data_profile.json, analysis_results.json'],
    ['교육용 업무 가정','30일, 20대, 300/600/60만원','README.txt 1·3절'],
    ['공식 방법 정의','AP·ROC·Brier·RF 확률·순열 중요도·전처리·보정곡선 API','계산식 대조표 S01–S18'],
    ['프로젝트 설계 판단','q99·90/180일·분할일·고정 후보·시드·개별 민감도','소스 함수·analysis_config.json'],
    ['미검증','2025 시험 성능·새 기종/부대·실제 예방 효과·실제 절감·인과','현재 결과로 확정하지 않음']
])}

기존 분석·앱의 검증은 [검증 기록](verification.md)에 있다. 이번 보고에서는 AP·사다리꼴 PR·ROC·Brier를 별도 식으로 재계산하고,12개 월/집계 행·무작위 기준선·비용표 전{evidence['cost_rows_recomputed']}행·앞 기준·개별 민감도·전체 순열 중요도{evidence['permutation_features_reproduced']}개를 대조했다. 오차 허용은1e-12이며 **원본·기존 모델·분석 결과의 해시는 변경되지 않았다**. 결과는 [독립 재검산 JSON](briefing_evidence.json), 문서/생성기 해시는 `briefing_manifest.json`이다.

```powershell
python scripts/build_briefing.py
python scripts/check.py --predictions reports/test_predictions_2025.csv
```

실제 적용 전에는 비용 가정·정비 개입 효과·확률 보정·다른 기간/장비/부대 검증과 자원·임무 중요도를 확인해야 한다. 모델의 재고장 가능성 순위가 종합 작전 위험도나 담당자의 판단을 대체하지 않는다. 수치·모델·설명 방식은 이 드론 실습에서 확인한 근거만 사용했다.
"""
    write_report("reports/2026-01_분석결과보고.md", text)


def build_speech(result, evidence):
    current, metric = result["current"], result["evaluation"]["metrics"]
    latest = [r for r in evidence["monthly_recomputed"] if r["unit"] == "latest_drone"]
    model_hits = sum(r["model_hits"] for r in latest)
    age_hits = sum(r["age_hits"] for r in latest)
    random_hits = sum(r["random_hits_mean"] for r in latest)
    example = current["equipment"][0]
    timing = table(["시간","화면과 조작","핵심"], [
        ["00:00–00:35","재점검 계획 상단","81대 중20대·위험22대·대기2대"],
        ["00:35–01:20","데이터·과정","중복·결측·상한·미래 정보·30일 경계"],
        ["01:20–02:15","모델 검증 비교·지표","RF 선택, AP/ROC 정의,38/18/13.8건"],
        ["02:15–03:20","재점검 계획 정렬→감시드론→돌아가기","전체 순위 유지·그룹 재배정 없음"],
        ["03:20–04:10","드론 상세 DR-0115","95.82%와 관측 이력·참조값 치환 설명"],
        ["04:10–05:00","모델 검증 비용·한계","가정 비용·임계값과 정원 분리·시험 채점 미확인"],
    ])
    sections = [
        ("00:00–00:35 · 결론과 성과", "재점검 계획 상단의20대·81대·22대·2대를 가리킨다.",
         f"지휘관님, 드론 예측정비 실습 결과를 보고드리겠습니다. 제공된2026년1월 정비106건을 최신 드론81대로 정리하고 재점검20대를 선정했습니다. 위험 기준 이상은22대여서 정원 밖2대는 추가 검토 대상으로 남겼습니다. 과거 하반기 월20대 선정의 적중은 모델{model_hits}건, 오래된 순{age_hits}건, 무작위 평균{random_hits:.1f}건입니다. 이는 가상 자료의 선정 성능이며 실제 고장을 예방한 건수는 아닙니다.",
         "analysis_results.json: current / evaluation.monthly(latest_drone). 산식F12·F13. 이번 달과 하반기 성능의 시점을 섞지 않는다."),
        ("00:35–01:20 · 데이터 신뢰성과 처리 이유", "데이터·과정의 원본 표·처리 설명·시간 경계를 보여준다.",
         "세 파일에서 완전 중복과 같은 드론·같은 날짜 중복은 없었습니다. 예방정비의 고장 공란과 관측 첫 정비의 이력 없음은 구분했습니다. 큰 시간·비용은 오류라고 단정하지 않고 행을 보존한 채 학습 상한 처리 대안을 비교했습니다. 월말 가동률은 정비 직후 가용성이 확인되지 않아 제외했습니다. 과거 기록만 피처로 쓰고,30일 정답이 확정된 학습 행만 다음 시점 예측에 사용했습니다.",
         "data_profile.json / splits / processing. 산식F01–F06. q99는 설계 선택이며 이 자료에는 999를 결측 또는 비용 100배를 오류로 정한 규칙이 없다."),
        ("01:20–02:15 · 모델과 평가 지표", "모델 검증에서 앞 기간 비교표와 뒤 평가 지표를 가리킨다.",
         f"앞 검증에서8개 모델·피처 구성을 비교해 Random Forest를 선택했습니다. 앞 AP는0.7158로 로지스틱0.6953,LightGBM0.6513보다 높았습니다. 선택에 사용하지 않은 하반기 AP는{metric['ap']:.4f},ROC-AUC는{metric['roc_auc']:.4f}입니다. AP는 재고장 사례를 높은 순위에 모으는 정도,ROC-AUC는 고장과 비고장의 순위 구분력입니다. 정확도78.77퍼센트라는 뜻은 아닙니다. 월20대 적중도 함께 보며 시험2025년 성능은 정답이 없어 보고하지 않습니다.",
         "comparisons / evaluation.metrics. 산식F07·F09–F13·F16. 앞 기간으로 선택, 뒤 기간으로 고정 평가했다."),
        ("02:15–03:20 · 순위와 요약 상세 시연", "아래 필수 클릭 표를 따라가고 클릭마다 응답을 기다린다.",
         "목록은 드론,유형,부대,최근 정비일,정비 구분,확률로 정렬할 수 있습니다. 부대 순으로 바꿔도 전체 위험순위와 선정 대상은 유지됩니다. 유형 요약의 감시드론 이름을 누르면 선정8대뿐 아니라 전체 후보20대를 확인합니다. 돌아가면 선정20대 목록과 요약이 복원됩니다. 정비 구분의 사후정비 전체18대도 같은 방식입니다. 그룹 안에서 정원을 새로 배분하는 기능은 아닙니다.",
         "current.equipment / app_helpers.py. 산식F12. 현재 앱은 구분 이름을 눌러 전체 상세를 여는 동작이다."),
        ("03:20–04:10 · 위험 요인과 가중치", "드론 상세에서DR-0115를 선택하고 요인 막대·값 표·과거 이력을 보여준다.",
         f"최상위 {example['equipment_id']}의 모델 추정 확률은 {example['prob']*100:.2f}퍼센트입니다. 이전 90일 정비 12건, 관측 고장 후 1일, 이전 90일 사후·긴급정비 10건이 주요 상승 설명입니다. 사람이 고장 횟수 40퍼센트 같은 가중치를 부여한 것은 아닙니다. 240개 트리 확률의 평균입니다. 설명 막대는 한 입력을 학습 참조값으로 바꿨을 때의 모델 확률 차이입니다. SHAP이나 실제 고장 원인, 합산 가중치로 해석하지 않습니다.",
         "current.equipment[0].factors / briefing_evidence.json. 산식F07·F15. %와%p를 구분한다."),
        ("04:10–05:00 · 임계값·한계·검토 사항", "모델 검증의 비용 표와 사이드바40.14%·정원20대를 가리킨다.",
         "위험 기준40.14퍼센트는 놓친 재고장300만원과 점검60만원이라는 과제 가정으로 앞 검증 비용을 최소화해 골랐습니다. 놓친 비용을600만원으로 높이면 기준은22.76퍼센트로 낮아집니다. 정원20대와 별도 설정이며 실제 절감액을 증명한 것은 아닙니다. 정원 밖 위험2대의 이력과 추가 점검 가능성을 검토하고,실제 비용·임무 중요도·개입 효과·다른 부대 검증을 확인해야 합니다. 최종 정비 판단은 담당자가 하도록 제안드립니다. 이상입니다.",
         "README.txt 비용 가정 / threshold_policies / evaluation.frozen_policy_costs. 산식F14. 검토 제안이지 실제 운용 승인 완료가 아니다."),
    ]
    text = f"""# 드론 예측정비 실습 결과 · 지휘관5분 핵심 발표 대본

- 과정: 군 특화 AI 활용 데이터분석 과정, rdm_ai_lab 실습 결과보고
- **5분 핵심 요약 + 앱 시연** 구성이다. 본문의 발표 대사만 낭독하고 근거·부록은 질의 시 사용한다.
- 수치 근거: [분석결과보고](2026-01_분석결과보고.md), [계산식·출처](계산식_출처_대조표.md), [재점검 계획](reinspection_plan_2026-01.md), [재검산 결과](briefing_evidence.json)
- 교육용 가상 자료, 대상월2026-01. 실제 현재 월이나 실제 부대 운용 성과로 말하지 않는다.

## 발표 전 · 기본 상태

```powershell
python scripts/build_briefing.py
python scripts/check.py --predictions reports/test_predictions_2025.csv
```

1. 실행 중 앱은 `http://127.0.0.1:8503`. 새로 시작할 때는 `python -m streamlit run app.py`로 실행하고 터미널에 표시된 주소를 쓴다. 발표 중 재학습하지 않는다.
2. **재점검 가능 대수 20**, **위험 표시 기준 직접 조정 끔**, 기준 **40.14%**인지 확인한다. 드론 유형·정비 구분 필터와 검색을 비우고 부대는 **전체**로 둔다.
3. **재점검 계획**에서 돌아가기 버튼이 보이면 눌러 요약을 복원한다. 목록 범위는 **선정 대상**, 정렬은 **확률·내림차순**으로 맞춘다.
4. 상단 **20대/81대/22대/2대**, 최상위 **DR-0115**, 확률 **95.82%**를 확인한다. 표 내부의 열 제목 추가 정렬과 위의 명시적 정렬을 구분한다.
5. 감시드론 요약 **전체20/선정8/위험8**, 사후정비 요약 **전체18/선정10/위험11**을 미리 확인한다. 구분 **이름**이 상세 버튼이다.
6. 모델 검증은 하반기 **AP 0.7877 / ROC-AUC 0.8814**, 월20대 **모델38/연수18/무작위13.8건**이다. **2025 시험 성능이라고 말하지 않는다.**
7. 원본 계획·계산식 대조표·이 대본을 미리 열고 클릭 순서를 리허설한다. 5분은 조작 응답 속도에 따라 조정하며 부록은 낭독하지 않는다.

## 5분 시간표

{timing}

"""
    for title, action, speech, source in sections:
        text += f"## {title}\n\n**화면 조작:** {action}\n\n**발표 대사**\n\n“{speech}”\n\n**근거·주의:** {source}\n\n"
    text += "## 필수 클릭 순서와 선택 시연\n\n" + table(["순서","구분","조작","확인"], [
        [1,"필수","정렬 속성 부대·오름차순","전체 순위·선정은 유지"],
        [2,"필수","확률·내림차순 복원","DR-0115가 최상위"],
        [3,"필수","유형별 요약 감시드론 이름 클릭","전체 후보20대, 선정8대 포함"],
        [4,"필수","요약으로 돌아가기","선정20대·정렬 설정 복원"],
        [5,"시간 여유 시","정비 구분별 사후정비 이름 클릭","전체18대, 선정10대 포함"],
        [6,"5번 수행 시 필수","요약으로 돌아가기","전체 요약·선정 목록 복원"],
        [7,"필수","드론 상세DR-0115","과거 이력과 참조값 치환 민감도 확인"],
    ])
    faqs = [
        ("중복은 없었는가?", "세 파일의 완전 중복·드론/정비일 중복 추가행은0입니다. 이번 달106건을81대로 줄인 것은18대의 반복 정비를 이력으로 보존하고 선정 시 최신 행만 쓴 결과입니다. F01·F12, data_profile.json."),
        ("999시간·19,000만원은 어떻게 처리했는가?", "이 자료에는999=미입력 또는 비용100배 오류라는 제공 규칙이 없습니다. 원본을 보존하고 q99 상한 대안을 비교했습니다. RF 자체의 raw 대조는 없어 상한이 RF를 개선했다고 단정하지 않습니다. F02, processing 및 비교표."),
        ("결측 선정 기준과 근거는?", "공란이 예방정비에만 위치하고 경과일 공란150건은 첫 관측 정비인 것을 확인했습니다. 해당 없음·이력 없음으로 구분한 가설이며 생성 원인 확정은 아닙니다. 중앙값은 학습 기간에서만 추정했습니다. F03."),
        ("가동률·식별자·코드를 왜 뺐는가?", "가동률은 시점·이번 달 열 부재 근거, ID는 암기 의존 제한 설계입니다. 부대·고장 코드는 묶음 추가 대안의 AP를 비교했으나 각각의 독립 효과는 확인하지 않았습니다. 분석보고4절."),
        ("요인과 가중치는 어떻게 정했는가?", "업무 시점에 알 수 있는 상태·과거 이력을 후보로 삼고8개 고정구성을 비교했습니다. RF는 Gini 분기를 학습하고240개 잎 확률을 평균합니다. 사람이 정한 요인별40/30% 가중치는 없습니다. F04·F07·F15."),
        ("다른 모델보다 나아서RF를 골랐는가?", "앞 검증 AP0.7158이 이번 고정 LR0.6953·LightGBM0.6513보다 높아 규칙대로 선택했습니다. 전 모델 최적 튜닝·통계적 유의성·다른 부대의 우위를 입증한 것은 아닙니다. 하반기는 재선정에 쓰지 않았습니다. 분석보고7절."),
        ("AP와 ROC-AUC는 정확도인가?", "아닙니다. AP는 재현율 증가로 가중한 정밀도, ROC-AUC는 양성·음성 순위 구분입니다. 동점은 ROC에서 0.5로 셉니다. AP와 사다리꼴 PR 면적도 계산이 달라 구분합니다. F09–F11."),
        ("왜 20대 적중 38건과 기록 적중 93건이 다른가?", "38건은 월별 최신 고유 드론 20대, 93건은 월별 정비 기록 20건이라 같은 드론 반복이 가능합니다. AP 643건·최신 497드론-월·선정 120건의 분모도 다릅니다. F12·F13, 분석보고 8절."),
        ("40.14%의 계산식과 출처는?", "과제의 비용 가정으로 FN×300+표시 건수×60을 앞 검증에서 최소화했습니다. 실제 대입은 40×300+111×60=18,660만원입니다. 600만원이면 27×600+182×60=27,120만원이고 기준은 22.76%입니다. F14; README.txt·threshold_costs.csv."),
        ("이론상 60/300은 20%인데 왜 40.14%인가?", "20%는 확률이 완전히 보정되고 점검으로 고장을 완전히 예방한다는 가정의 기대 손실 분기점입니다. 실제 사용값은 검증 예측·정답에서 경험 비용을 최소화한 값입니다. 보정과 재학습 후 기준을 별도 확인해야 합니다. F14."),
        ("설명 막대 16.7%p를 가중치나 예방 효과로 봐도 되는가?", "한 입력을 학습 참조값으로 치환했을 때 모델 출력이 달라진 값입니다. 합산 기여·SHAP·인과·예방 효과가 아닙니다. 전체 순열 중요도의 AP 감소와도 다른 척도입니다. F15."),
        ("정원 밖 위험 2대와 시험 성능은?", "위험하다고 표시한 드론을 정원 때문에 정상으로 바꾸지 않습니다. 대기·운용 전 확인·추가 점검은 정비반장 판단입니다. 시험·이번 달 정답이 없어 실제 성능은 알 수 없고 강사 채점을 기다려야 합니다. current/test 및 한계."),
    ]
    text += "\n\n## 질의응답 · 계산식과 출처로 답하기\n\n"
    for index, (question, answer) in enumerate(faqs, 1):
        text += f"### {index}. {question}\n\n{answer}\n\n"
    text += """## 질의 때만 임계값·정원 추가 시연

1. 기본 20대·위험 22대·임계값 직접 조정 끔을 확인한다.
2. 직접 조정을 켜 기준 100%로 바꾸면 위험 표시는 0대지만 선정 20대는 유지된다.
3. 정원을 5대로 바꾸면 선정 5대가 되며 확률 순위는 유지된다.
4. 정원 20대로 복원하고 직접 조정을 끄면 위험 22대·대기 2대가 돌아온다. 필터와 목록 범위도 초기화한다.

## 시연 실패·시간 초과 대응

| 상황 | 조치 | 이어갈 문장 |
| --- | --- | --- |
| 기본 지표가 다름 | 직접 조정·정원·목록 범위·필터 확인. 10초 안에 복원되지 않으면 정적 계획으로 전환 | “검증한 기본 계획으로 수치를 보고드리겠습니다.” |
| 구분 클릭 지연 | 응답을 기다리고 반복 클릭하지 않음. 정적 요약표로 대체 | “전체 후보와 선정 명단을 구분해 확인하는 절차입니다.” |
| 정렬 혼동 | 위 정렬 설정과 표 열 제목의 추가 정렬을 구분 | “보기 순서와 전체 위험순위는 별개입니다.” |
| 근거 파일·앱 오류 | 발표 중 재학습하지 않고 저장 보고서·재검산 JSON 사용 | “현재 화면 연결과 계산 결과의 근거를 구분해 보고드리겠습니다.” |
| 시간 초과 | 사후정비 상세 클릭과 전체 중요도 수치 낭독 생략 | “비교 결과와 적용 한계를 중심으로 말씀드리겠습니다.” |

## 발표자료의 참고 출처

정의·수식·코드·숫자 필드가 연결된 [계산식·출처 대조표](계산식_출처_대조표.md)와 [공식 문서 목록](method_sources.json)을 사용한다. 과제 가정은 README.txt, 결과는 analysis_results.json, 독립 수식 재검산은 briefing_evidence.json이다.
"""
    write_report("reports/2026-01_지휘관발표스크립트.md", text)


if __name__ == "__main__":
    main()
