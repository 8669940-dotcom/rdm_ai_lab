"""Generate report-ready process evidence from executed outputs, never inferred test labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from rdm_model import FEATURE_LABELS, TARGET, load_data
from scripts.rdm_common import ROOT, sha256


def md_table(columns, rows):
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")
    return ["| " + " | ".join(map(cell, columns)) + " |", "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(map(cell, row)) + " |" for row in rows]]


def main():
    reports = ROOT / "reports"
    result = json.loads((reports / "analysis_results.json").read_text(encoding="utf-8"))
    raw = load_data()
    train = raw[raw["_source"].eq("train")].copy()
    train["정비 월"] = train["maintenance_date"].dt.month.astype(str).str.zfill(2)
    train["fault_category"] = train["fault_category"].fillna("N/A")
    eda = []
    for field, label in [("equipment_type", "드론 유형"), ("정비 월", "정비 월"), ("fault_category", "고장 유형")]:
        for category, group in train.groupby(field, sort=True):
            eda.append({"기준": label, "구분": category, "정비 건수": len(group),
                        "재고장 건수": int(group[TARGET].sum()), "재고장률": float(group[TARGET].mean())})
    pd.DataFrame(eda).to_csv(reports / "eda_summary.csv", index=False, encoding="utf-8-sig")
    metric, current, processing = result["evaluation"]["metrics"], result["current"], result["processing"]
    latest = [x for x in result["evaluation"]["monthly"] if x["unit"] == "latest_drone"]
    lines = [
        "# 드론 예측정비 분석 과정·근거 기록", "",
        "2026-11-17 사용자 지정 N/A·비용 오류 가정으로 재분석한 현재 결과다. 과거 q99 결과는 Git 이력과 진행 기록에 구분한다.", "",
        "## Ⅰ. AI 활용", "",
        "사람이 목표·비용·정원·999와 비용 오류 가정을 제시했고, Codex가 데이터 점검·코드·앱·보고서·검증 작성에 활용됐다.",
        "숫자는 Python 실행 결과에서 가져오며 시험 정답을 생성·복원하지 않았다. 가정과 최종 정비 판단은 사람이 확인한다.",
        "상세 최종 문서: [분석결과보고](2026-01_분석결과보고.md), [5분 대본](2026-01_지휘관발표스크립트.md).", "",
        "## Ⅱ. 데이터 분석", "", "### 1. 입력·보존", "",
        "학습 2,346건·시험 1,241건·이번 달 106건. 완전/드론·날짜 중복은 0. 원본 SHA-256은 data_manifest.json과 대조한다.",
        "원본·행은 삭제하지 않는다. 이번 달은 최신 81대에서 20대를 선정한다. 학습 외 정답은 없다.", "",
        "### 2. 전처리", "",
        "시간 999·숫자 공란은 N/A로 보고 학습 중앙값과 결측 표시를 쓴다. 범주 공란은 N/A 범주다.",
        "비용은 학습 1.5 IQR 울타리 밖만 후보로 삼는다. 100×Q3 초과는 100배 오류 가정으로 나눈 뒤 Q1~Q3로 제한하고, 나머지 후보는 직접 Q1~Q3로 제한한다.",
        "유효 비용은 유지한다. 오류가 확인된 사실로 쓰지 않는다. 다른 분할·시험·이번 달 분포로 통계량을 학습하지 않는다.",
        "치환 전후·원본 행·모델 입력은 [감사 CSV](cleaning_audit.csv), 학습 범위·처리량은 analysis_results.json: processing에 있다.", ""]
    lines += md_table(["역할", "Q1", "Q3", "하단 울타리", "상단 울타리"], [
        [name, x["q1"], x["q3"], x["lower_fence"], x["upper_fence"]]
        for name, x in processing["fitted_by_role"].items()])
    lines += ["", "### 3. 피처·시간 분할", "",
              "최종 한글 피처: " + ", ".join(processing["feature_labels"]) + ".",
              "ID는 이력·제출 식별자, 날짜는 분할·월 주기·이력에 사용한다. 월말 가동률은 정비 직후 가용성 미확인과 이번 달 열 부재로 제외한다.",
              "부대·고장 코드는 묶음 추가 대안도 비교했다. 개별 열의 독립 효과를 분리 검증한 것은 아니다.",
              "현재 정비일보다 앞선 기록만 이력으로 사용한다. 정비일+30일 < 예측 시작일인 학습 행만 사용한다.",
              "앞 검증은 2024-01~05, 뒤 검증은 2024-07~12다. 뒤 구간으로 재선정하지 않는다.",
              "이 하반기는 이전 분석에서도 확인했으므로 재분석 전체의 새로운 외부 검증이라고 주장하지 않는다.", "",
              "### 4. 모델·지표·기준선", "",
              f"선택 구성: {result['selected_model_label']}. 4알고리즘×3피처 묶음 모두 동일한 새 전처리를 사용한다.",
              f"하반기 AP {metric['ap']:.6f}, 사다리꼴 PR 면적 {metric['pr_auc_trapezoid']:.6f}, ROC-AUC {metric['roc_auc']:.6f}, Brier {metric['brier']:.6f}.",
              "AP=ΣRecall 증가량×Precision. Precision=TP/(TP+FP), Recall=TP/(TP+FN), True Positive Rate (TPR)=Recall, False Positive Rate (FPR)=FP/(FP+TN).",
              f"최신 드론 월20대 적중 합 {sum(x['model_hits'] for x in latest)}건, 연수 기준 {sum(x['age_hits'] for x in latest)}건, 무작위 1000회 평균 합 {sum(x['random_hits_mean'] for x in latest):.3f}건.",
              "같은 월·후보에서 비교하며 정비 기록 상위20건은 별도 집계다. 정확도 하나로 성능을 판단하지 않는다.", "",
              "### 5. 비용·현재 계획", ""]
    lines += md_table(["가정(만원)", "기준", "FN", "표시 건", "비용(만원)"], [
        [cost, p["threshold"], p["false_negatives"], p["flagged"], p["cost"]]
        for cost, p in result["threshold_policies"].items()])
    lines += ["", "비용=FN×300(또는600)+위험 표시 건수×60. 앞 검증에서 최소화하고 시험으로 고르지 않는다.",
              f"이번 달 선정 {current['selected']}대, 위험 {current['risk_flagged']}대, 정원 밖 위험 {current['risk_outside_capacity']}대. 정원과 기준은 독립이다.",
              "개별 설명은 학습 참조값 치환 확률 차이, 전체 중요도는 검증 AP 순열 감소다. 가중치·인과·실제 예방 효과와 구분한다.", "",
              "### 6. 재현·산출물·한계", "",
              "python scripts/train.py → python scripts/build_report.py → python scripts/build_briefing.py → python scripts/check.py --predictions reports/test_predictions_2025.csv.",
              "시험 예측은 원본 1,241행·3열·유한 확률·키 보존을 검사한다. 2025 성능은 정답이 없어 산출하지 않는다.",
              "분석값은 analysis_results.json, 코드·데이터 해시와 14개 주요 산출물 해시는 output_manifest.json, 문서 해시는 briefing_manifest.json이다.",
              "가상 데이터, 미검증 오류 가정·실제 고비용 정보 손실·다른 부대 일반화·확률 보정·실제 개입 효과는 한계다.",
              "진행/문제 해결은 preparation_log.md, 실제 검사 결과는 check_result.json, 공식 출처와 계산식은 계산식_출처_대조표.md를 참조한다.", ""]
    (reports / "analysis_process.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    plan = ["# 2026년 1월 드론 재점검 계획", "",
            f"최신 {current['candidates']}대 중 {current['selected']}대. 위험 {current['risk_flagged']}대·정원 밖 위험 {current['risk_outside_capacity']}대.",
            f"선택 구성: {result['selected_model_label']}. 확률순, 동점은 드론 번호순.", ""]
    plan += md_table(["순위", "드론", "유형", "부대", "최근 정비", "정비 구분", "확률", "상승 요인", "권고"], [
        [x["rank"], x["equipment_id"], x["equipment_type"], x["unit_code"], x["maintenance_date"], x["maintenance_type"],
         f"{x['prob']:.2%}", x["reason"], x["recommendation"]] for x in current["equipment"] if x["selected"]])
    waiting = ", ".join(f"{x['equipment_id']} ({x['prob']:.2%})" for x in current["equipment"] if x["risk_flag"] and not x["selected"]) or "없음"
    plan += ["", "정원 밖 위험 대기: " + waiting, "",
             "정비반장이 대기·운용 전 확인·추가 점검 가능성을 검토한다. 설명은 참조값 치환 민감도이며 가중치·인과·예방 효과가 아니다.",
             "교육용 가상 자료이며 실제 예방 효과와 외부 일반화는 검증하지 않았다.", ""]
    (reports / "reinspection_plan_2026-01.md").write_text("\n".join(plan), encoding="utf-8", newline="\n")
    outputs = ["reports/analysis_results.json", "reports/test_predictions_2025.csv", "reports/reinspection_plan_2026-01.csv",
               "reports/current_candidates_2026-01.csv", "reports/validation_predictions_2024H2.csv", "reports/monthly_validation.csv",
               "reports/threshold_costs.csv", "reports/eda_summary.csv", "reports/analysis_process.md", "reports/reinspection_plan_2026-01.md",
               "reports/cleaning_audit.csv", "artifacts/rdm_model.joblib", "artifacts/submission_model.joblib", "artifacts/model_card.json"]
    (reports / "output_manifest.json").write_text(json.dumps({p: sha256(ROOT / p) for p in outputs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Saved EDA, process record, plan and 14 output hashes.")


if __name__ == "__main__":
    main()
