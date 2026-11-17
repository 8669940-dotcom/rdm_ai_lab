"""Drone maintenance planning dashboard. No training or external writes on rerun."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app_helpers import (DISPLAY, GROUP_FIELDS, SORT_FIELDS, export_csv, filter_frame,
                         group_summary, plan_frame, sorted_frame)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RDM | 드론 재점검 계획", page_icon=":material/flight:", layout="wide")
st.caption("RDM  /  군 특화 AI 활용 데이터분석 실습")
st.title("드론 재점검 의사결정")
st.caption("2026년 1월 월말 스냅샷 · 교육용 가상 데이터 · 정비 직후 30일 이내 재고장 예측")


@st.cache_data(max_entries=2, show_spinner=False)
def load_outputs(version):
    results = json.loads((ROOT / "reports/analysis_results.json").read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "reports/data_profile.json").read_text(encoding="utf-8"))
    costs = pd.read_csv(ROOT / "reports/threshold_costs.csv")
    frames = []
    for name in ("drone_train.csv", "drone_test.csv", "drone_this_month.csv"):
        frame = pd.read_csv(ROOT / "data" / name, encoding="utf-8-sig")
        frame["원본 파일"] = name
        frames.append(frame)
    eda_path = ROOT / "reports/eda_summary.csv"
    eda = pd.read_csv(eda_path, encoding="utf-8-sig") if eda_path.exists() else pd.DataFrame()
    return results, profile, costs, pd.concat(frames, ignore_index=True), eda


result_path = ROOT / "reports/analysis_results.json"
if not result_path.exists():
    st.info("분석 산출물을 먼저 생성하세요: python scripts/train.py")
    st.stop()
results, profile, costs, raw, eda = load_outputs((result_path.stat().st_mtime_ns,
                                                (ROOT / "reports/eda_summary.csv").stat().st_mtime_ns if (ROOT / "reports/eda_summary.csv").exists() else 0))
base_policy = results["threshold_policies"]["300"]
base_threshold = base_policy["threshold"]
st.session_state.setdefault("group_filter", None)
st.session_state.setdefault("return_scope", "선정 대상")
st.session_state.setdefault("list_scope", "선정 대상")

with st.sidebar:
    st.header("운영 조건")
    st.caption("기본 계획: 81대 중 20대. 변경값은 화면 시나리오에만 적용됩니다.")
    capacity = int(st.number_input("재점검 가능 대수", min_value=1, max_value=results["current"]["candidates"],
                                   value=results["config"]["capacity"], key="capacity", persist_state="session"))
    manual = st.toggle("위험 표시 기준 직접 조정", value=False, key="manual_threshold", persist_state="session")
    threshold = base_threshold
    if manual:
        threshold = st.slider("위험 표시 기준 (%)", 0., 100., round(float(base_threshold if base_threshold is not None else 1.) * 100, 1), .1,
                              key="threshold_percent", persist_state="session") / 100
    threshold_text = f"{threshold:.2%}" if threshold is not None else "위험 표시 없음"
    st.caption(f"현재 기준 {threshold_text} · 검증 미탐지 비용 300만원 시나리오")
    st.divider()
    st.subheader("목록 필터")
    types = st.pills("드론 유형", sorted({r["equipment_type"] for r in results["current"]["equipment"]}),
                     selection_mode="multi", key="types", persist_state="session")
    unit = st.selectbox("부대", ["전체"] + sorted({r["unit_code"] for r in results["current"]["equipment"]}),
                        key="unit", persist_state="session")
    maintenance = st.pills("정비 구분", ["예방정비", "사후정비", "긴급정비"], selection_mode="multi",
                           key="maintenance", persist_state="session")
    query = st.text_input("드론 번호 검색", placeholder="예: DR-0002", key="search", persist_state="session")
    st.caption("필터는 전체 순위와 선정 명단을 바꾸지 않습니다.")

plan = plan_frame(results, capacity, threshold)
outside = plan[plan["risk_flag"] & ~plan["selected"]]
scenario = manual or capacity != results["config"]["capacity"]
with st.container(horizontal=True):
    st.metric("재점검 선정", f"{int(plan['selected'].sum())}대", border=True)
    st.metric("전체 후보", f"{len(plan)}대", border=True)
    st.metric("위험 기준 이상", f"{int(plan['risk_flag'].sum())}대", border=True)
    st.metric("위험 초과 대기", f"{len(outside)}대", border=True)
if scenario:
    st.info("운영 시나리오를 보고 있습니다. 저장된 공식 20대 계획과 학습 결과는 유지됩니다.")
page = st.segmented_control("화면", ["재점검 계획", "모델 검증", "데이터·과정"], default="재점검 계획", key="view") or "재점검 계획"


def open_group(column, click_key):
    click = st.session_state.get(click_key)
    if click:
        st.session_state["return_scope"] = st.session_state.get("list_scope", "선정 대상")
        st.session_state["group_filter"] = {"column": column, "value": click["label"]}


def back_to_summary():
    st.session_state["group_filter"] = None
    st.session_state["list_scope"] = st.session_state.get("return_scope", "선정 대상")


def open_equipment():
    click = st.session_state.get("equipment_click")
    if click:
        st.session_state["detail_id"] = click["label"]


if page == "재점검 계획":
    st.subheader("확률 순위로 정하고, 관측 근거로 점검합니다")
    if len(outside):
        st.warning(f"위험 기준 이상 {int(plan['risk_flag'].sum())}대 중 {len(outside)}대가 정원 밖입니다. "
                   "확률 순 대기 명단으로 관리하고 정비반장이 추가 점검·운용 전 확인 여부를 판단하세요.")
    else:
        st.caption("위험 표시와 재점검 정원은 독립적입니다. 위험 표시가 없어도 순위에 따라 정원을 선정합니다.")
    base_filtered = filter_frame(plan, types, unit, maintenance, query)
    group = st.session_state["group_filter"]
    if group:
        st.button("요약으로 돌아가기", icon=":material/arrow_back:", key="back_group", on_click=back_to_summary)
        st.subheader(f"{group['value']} · 전체 후보 상세")
        shown = filter_frame(base_filtered, group=group)
    else:
        with st.container(border=True):
            summary_kind = st.segmented_control("요약 기준", list(GROUP_FIELDS), default="유형별", key="summary_kind",
                                                persist_state="session")
            group_column = GROUP_FIELDS[summary_kind or "유형별"]
            summary = group_summary(base_filtered, group_column)
            chart_data = summary.rename(columns={group_column: "구분"}).melt(id_vars="구분", value_vars=["후보", "선정", "위험표시"],
                                                                           var_name="집계", value_name="대수")
            st.altair_chart(alt.Chart(chart_data, mark="bar").encode(
                x=alt.X("구분:N", title=None), y=alt.Y("대수:Q", title="드론 수"),
                color="집계:N", xOffset="집계:N", tooltip=["구분", "집계", "대수"]
            ).properties(height=220), alt="유형·부대·정비 구분별 후보와 선정 및 위험 표시 대수")
            st.dataframe(summary, hide_index=True, column_config={
                group_column: st.column_config.ButtonColumn("구분을 눌러 상세 보기", type="tertiary",
                                                            on_click=open_group, args=(group_column, "group_click"), key="group_click"),
                "평균확률": st.column_config.ProgressColumn("평균 확률", format="percent", min_value=0., max_value=1.),
            }, alt="구분 이름을 누르면 해당 범주의 전체 후보를 보여주는 요약표")
        scope = st.segmented_control("목록 범위", ["선정 대상", "전체 후보", "위험 초과 대기"],
                                     key="list_scope", persist_state="session") or "선정 대상"
        shown = base_filtered
        if scope == "선정 대상":
            shown = shown[shown["selected"]]
        elif scope == "위험 초과 대기":
            shown = shown[shown["risk_flag"] & ~shown["selected"]]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        sort_field = st.selectbox("정렬 속성", list(SORT_FIELDS), key="sort_field", persist_state="session")
        direction = st.segmented_control("정렬 방향", ["내림차순", "오름차순"], default="내림차순", key="sort_direction",
                                         persist_state="session")
    shown = sorted_frame(shown, sort_field, direction == "오름차순")
    st.caption(f"현재 표시 {len(shown)}대 · 전체 후보에서 선정한 순위 유지 · 열 제목으로도 정렬할 수 있습니다.")
    columns = ["rank", "equipment_id", "equipment_type", "unit_code", "maintenance_date", "maintenance_type", "prob", "selected", "risk_flag"]
    table = shown[columns].rename(columns=DISPLAY)
    st.dataframe(table, hide_index=True, height=390, column_config={
        "드론": st.column_config.ButtonColumn("드론 상세", on_click=open_equipment, key="equipment_click", type="tertiary"),
        "재고장 확률": st.column_config.ProgressColumn(format="percent", min_value=0., max_value=1.),
    }, alt="최신 정비 기록 기준 드론별 확률·전체 순위·선정·위험 표시 목록")
    with st.container(horizontal=True):
        st.download_button("현재 표시 목록 CSV", export_csv(shown), file_name="rdm_current_view.csv", mime="text/csv", key="download_view")
        st.download_button("공식 20대 계획 CSV", (ROOT / "reports/reinspection_plan_2026-01.csv").read_bytes(),
                           file_name="reinspection_plan_2026-01.csv", mime="text/csv", key="download_plan")
    if shown.empty:
        st.info("조건에 맞는 드론이 없습니다. 목록 범위나 필터를 변경하세요.")
    else:
        ids = shown["equipment_id"].tolist()
        if st.session_state.get("detail_id") not in ids:
            st.session_state["detail_id"] = ids[0]
        chosen_id = st.selectbox("드론 상세", ids, key="detail_id", persist_state="session")
        detail = plan[plan["equipment_id"].eq(chosen_id)].iloc[0]
        with st.container(border=True):
            st.subheader(f"{chosen_id} · {detail['equipment_type']} · {detail['unit_code']}")
            st.caption(f"최근 정비 {detail['maintenance_date']} / {detail['maintenance_type']} / 전체 {int(detail['rank'])}위")
            st.metric("30일 이내 재고장 예측 확률", f"{detail['prob']:.2%}")
            factors = pd.DataFrame(detail["factors"])
            if not factors.empty:
                factors["확률 차이(%p)"] = factors["delta_probability"] * 100
                st.bar_chart(factors, x="label", y="확률 차이(%p)", horizontal=True,
                             x_label="관측 특성", alt="각 특성을 학습 참조값으로 바꿀 때 줄어드는 예측 확률")
                st.dataframe(factors[["label", "value", "reference", "확률 차이(%p)"]].astype({"value": str, "reference": str})
                             .rename(columns={"label": "특성", "value": "관측값", "reference": "학습 참조값"}),
                             hide_index=True, alt="위험 상승 특성의 관측값과 학습 중앙값·최빈값 비교")
            st.caption("한 특성씩 학습 참조값으로 치환한 확률 민감도입니다. 합산 기여도·인과관계·실제 예방 효과를 뜻하지 않습니다.")
            st.info(str(detail["recommendation"]))
            history = raw[raw["equipment_id"].eq(chosen_id) & raw["maintenance_date"].le(detail["maintenance_date"])]
            st.dataframe(history[["maintenance_date", "maintenance_type", "fault_category", "severity", "maintenance_success", "원본 파일"]]
                         .sort_values("maintenance_date", ascending=False).rename(columns={"maintenance_date": "정비일", "maintenance_type": "정비 구분",
                                                                                           "fault_category": "고장 유형", "severity": "심각도", "maintenance_success": "정비 성공"}),
                         hide_index=True, height=230, alt="현재 정비일까지의 해당 드론 관측 정비 이력")
    st.caption("가상 데이터 실습 결과입니다. 최종 점검·운용 판단은 정비 담당자가 수행합니다.")

elif page == "모델 검증":
    st.subheader("앞 기간에서 선택하고, 뒤 기간에서 평가했습니다")
    st.caption(f"모델 선택 2024년 1월부터 5월 → 추가 평가 2024년 7월부터 12월 · 선택 모델 {results['selected_model']} · 분할 경계 30일 정답 성숙 적용")
    evaluation = results["evaluation"]
    metric = evaluation["metrics"]
    monthly = pd.DataFrame(evaluation["monthly"])
    latest = monthly[monthly["unit"].eq("latest_drone")]
    with st.container(horizontal=True):
        st.metric("하반기 AP", f"{metric['ap']:.4f}", border=True)
        st.metric("하반기 ROC-AUC", f"{metric['roc_auc']:.4f}", border=True)
        st.metric("월 20대 적중 합계", f"{int(latest['model_hits'].sum())}건", border=True)
        st.metric("무작위 기준선 평균", f"{latest['random_hits_mean'].sum():.1f}건", border=True)
    st.markdown("**AP**는 재고장 사례를 높은 순위에 모으는 정도를 정밀도·재현율로 요약합니다. "
                "**ROC-AUC**는 재고장과 비재고장 간 확률 순위 구분력을 보여줍니다. 정원 20대의 실제 적중 수와 함께 판단합니다.")
    left, right = st.columns(2)
    with left:
        pr = pd.DataFrame(evaluation["pr_curve"]).assign(order=lambda f: range(len(f)))
        st.altair_chart(alt.Chart(pr, mark="line").encode(x=alt.X("recall:Q", title="재현율", scale=alt.Scale(domain=[0, 1])),
                                                       y=alt.Y("precision:Q", title="정밀도", scale=alt.Scale(domain=[0, 1])), order="order:Q")
                        .properties(height=230), alt="하반기 검증의 정밀도·재현율 곡선")
    with right:
        st.altair_chart(alt.Chart(pd.DataFrame(evaluation["roc_curve"]), mark="line").encode(
            x=alt.X("fpr:Q", title="거짓 양성률", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="참 양성률", scale=alt.Scale(domain=[0, 1]))).properties(height=230), alt="하반기 검증의 ROC 곡선")
    st.caption(f"AP={metric['ap']:.4f}, 사다리꼴 PR-AUC={metric['pr_auc_trapezoid']:.4f}로 계산 정의가 다릅니다. "
               f"양성률={metric['prevalence']:.2%}, Brier={metric['brier']:.4f} (낮을수록 확률 오차가 작음).")
    st.subheader("모델 비교 · 앞 검증 구간")
    comparison = pd.DataFrame(results["comparisons"])[["key", "ap", "roc_auc", "brier", "latest_top20_hits"]]
    comparison["선택"] = comparison["key"].eq(results["selected_model"])
    st.dataframe(comparison.sort_values("ap", ascending=False).rename(columns={"key": "모델·피처 구성", "ap": "AP", "roc_auc": "ROC-AUC",
                                                                                           "brier": "Brier", "latest_top20_hits": "월20대 적중 합"}),
                 hide_index=True, alt="앞 기간의 8개 모델·피처 구성 비교; 하반기 성능으로 재선정하지 않음")
    st.caption("AP 순으로 선택하고 동률이면 월20대 적중, Brier, 모델 키 순으로 결정했습니다. 모델의 일반적 우위를 의미하지 않습니다.")
    st.subheader("월별 20대와 기준선")
    st.dataframe(latest[["month", "candidates", "positives", "model_hits", "age_hits", "random_hits_mean"]]
                 .rename(columns={"month": "월", "candidates": "후보 드론", "positives": "재고장", "model_hits": "모델 적중",
                                  "age_hits": "경과연수 기준", "random_hits_mean": "무작위 평균"}),
                 hide_index=True, alt="고유 드론 최신 기록 기준 월20대 적중과 동일 후보 집단의 기준선 비교")
    st.caption("같은 드론이 다른 월에 재선정될 수 있어 합계 단위는 월별 선정 사건입니다. 무작위는 월별 1,000회 반복이며 정비 기록별 상위20건은 다운로드 자료에 별도 보관했습니다.")
    st.download_button("월별 두 평가 단위 CSV", (ROOT / "reports/monthly_validation.csv").read_bytes(),
                       file_name="monthly_validation.csv", mime="text/csv", key="download_monthly")
    st.subheader("위험 표시 비용 · 20대 정원과 별도")
    visible_costs = costs[costs["policy"].eq("threshold")].assign(미탐지비용=lambda f: f["miss_cost"].astype(str) + "만원")
    st.altair_chart(alt.Chart(visible_costs, mark="line").encode(
        x=alt.X("threshold:Q", title="위험 임계값"), y=alt.Y("cost:Q", title="가정 비용(만원)"),
        color="미탐지비용:N", tooltip=["threshold", "cost", "flagged", "false_negatives", "미탐지비용"]
    ).properties(height=230), alt="앞 검증에서 임계값을 바꿀 때 미탐지 300만원·600만원 가정의 비용")
    policy_rows = [{"미탐지 비용(만원)": int(key), "선택 임계값": policy["threshold"], "앞 검증 비용(만원)": policy["cost"],
                    "뒤 검증 비용(만원)": evaluation["frozen_policy_costs"][key]["cost"]}
                   for key, policy in results["threshold_policies"].items()]
    st.dataframe(pd.DataFrame(policy_rows), hide_index=True, alt="앞 검증에서 정한 임계값과 뒤 검증에 고정 적용한 비용")
    st.caption("비용=놓친 재고장×미탐지 비용+위험 표시 건수×60만원. 표시된 재고장을 예방한다고 가정한 비교이며 실제 예산 절감 효과가 아닙니다. 재학습 후 확률 척도 변화에 대한 추가 보정이 필요할 수 있습니다.")
    st.info("2025년 시험 파일에는 정답이 없습니다. 시험 PR-AUC·월20대 적중은 강사 채점 후 확인할 수 있습니다.")
    st.download_button("시험 예측 1,241건 CSV", (ROOT / "reports/test_predictions_2025.csv").read_bytes(),
                       file_name="test_predictions_2025.csv", mime="text/csv", key="download_test")

else:
    st.subheader("원본에서 판단까지, 계산 근거를 남겼습니다")
    quality = [{"파일": name, "정비 건수": item["rows"], "고유 드론": item["unique_equipment"],
                "열 수": item["column_count"], "완전 중복": item["exact_duplicate_extras"],
                "드론·날짜 중복": item["equipment_date_duplicate_extras"]} for name, item in profile["files"].items()]
    st.dataframe(pd.DataFrame(quality), hide_index=True, alt="원본별 행·드론·열 수와 중복 점검 결과")
    if not eda.empty:
        pattern_kind = st.segmented_control("학습 데이터 분포", ["드론 유형", "정비 월", "고장 유형"], default="드론 유형", key="pattern_kind") or "드론 유형"
        pattern = eda[eda["기준"].eq(pattern_kind)]
        st.altair_chart(alt.Chart(pattern, mark="bar").encode(
            x=alt.X("구분:N", title=pattern_kind), y=alt.Y("재고장률:Q", axis=alt.Axis(format=".0%")),
            tooltip=["구분", "정비 건수", "재고장 건수", alt.Tooltip("재고장률:Q", format=".2%")]
        ).properties(height=220), alt="2023~2024 학습 데이터의 드론 유형·월·고장 유형별 관측 재고장률")
        st.caption("2023~2024 전체 학습 기록의 기술 통계이며 시험 성능이나 인과 효과가 아닙니다. 분모가 작은 범주와 반복 관측에 유의하세요.")
    st.markdown("**결측 처리**: 예방정비의 고장 공란은 ‘해당 없음’ 후보로 구분하고, 관측 첫 정비의 경과일 공란은 이력 없음 표시와 학습 중앙값을 사용합니다. "
                "**극단값**: 원본 행을 유지하며 운용중단 시간·정비 비용의 학습 99백분위 상한 처리 대안을 비교했습니다.")
    excluded_reasons = {"equipment_id": "이력 결합·제출 식별자로 사용하고 모델의 ID 암기를 방지",
                        "maintenance_date": "원 날짜는 제외하고 정렬·월 주기 피처로 사용",
                        "availability_rate": "월말 정보의 정비 직후 가용성 미확인·이번 달 열 부재",
                        "failure_within_30days": "예측할 정답이므로 입력에서 제외",
                        "unit_code": "부대·코드를 묶어 추가한 대안의 앞 검증 AP가 낮아 제외",
                        "fault_code": "부대·코드 포함 대안의 앞 검증 AP가 낮아 최종 구성에서 제외"}
    st.dataframe(pd.DataFrame([{"항목": k, "근거": excluded_reasons.get(k, v)} for k, v in results["processing"]["excluded"].items()]),
                 hide_index=True, alt="최종 모델에서 제외한 컬럼과 사유")
    st.subheader("시간 경계와 학습량")
    split_labels = {"development_train": "모델 선택용 학습", "evaluation_train": "하반기 평가용 학습",
                    "submission_train": "2025 시험 제출용 학습", "operation_train": "2026년 1월 운영용 학습"}
    st.dataframe(pd.DataFrame([{"역할": split_labels[name], **info} for name, info in results["splits"].items()])
                 .rename(columns={"rows": "학습 건수", "start": "학습 시작", "end": "학습 마지막",
                                  "prediction_start": "예측 시작", "latest_label_end": "마지막 정답 관측 종료"}),
                 hide_index=True, alt="모델별 학습 기간·행 수·최종 정답 관측일·예측 시작일")
    st.caption("시험 기간·이번 달 이력은 앞 파일과 이어 계산하되 현재 정비일 이전 기록만 사용합니다. 미래 기록과 숨겨진 정답은 피처에 포함하지 않았습니다.")
    st.subheader("검증 구간의 중요 특성")
    importance = pd.DataFrame(results["permutation_importance"]).head(10)
    st.bar_chart(importance, x="label", y="ap_decrease_mean", horizontal=True,
                 x_label="특성", y_label="섞었을 때 AP 감소", alt="하반기 검증에서 특성을 섞었을 때의 AP 감소량 상위10개")
    st.caption("전체 모델의 검증 중요도이며 개별 드론의 위험 상승 설명과 다릅니다. 상관된 특성은 중요도가 나뉘거나 작게 보일 수 있습니다.")
    st.subheader("분석 진행 순서")
    st.markdown("1. README·데이터 계약 검토와 원본 해시 보존\n2. 결측·중복·극단값 점검과 과거 이력 피처\n"
                "3. 앞 검증 모델·임계값 선택과 뒤 검증 평가\n4. 시험 전량 예측과 이번 달 20대·위험 요인\n5. 앱의 같은 산출물 표시와 재현 검사")
    for title, filename in [("README 검토 기록", "README_review.md"), ("분석 진행 기록", "preparation_log.md"),
                            ("보고서 작성용 근거 기록", "analysis_process.md")]:
        path = ROOT / "reports" / filename
        if path.exists():
            st.download_button(title, path.read_bytes(), file_name=filename, mime="text/markdown", key="download_" + filename)
    st.caption("교육용 가상 데이터이며 새 기종·부대에 대한 일반화와 실제 예방 효과는 검증하지 않았습니다. 확률은 담당자의 점검 판단을 지원합니다.")
