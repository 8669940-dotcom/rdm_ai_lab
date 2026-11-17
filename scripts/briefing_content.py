"""Dynamic Korean report content; every reported result comes from executed JSON."""
from __future__ import annotations

from rdm_model import FEATURE_GROUPS, FEATURE_LABELS, MODEL_LABELS


def documents(r, profile, a, table, src, code):
    metric, cur, policies = r["evaluation"]["metrics"], r["current"], r["threshold_policies"]
    latest = [x for x in r["evaluation"]["monthly"] if x["unit"] == "latest_drone"]
    records = [x for x in r["evaluation"]["monthly"] if x["unit"] == "maintenance_record"]
    total = lambda field: sum(x[field] for x in latest)
    p300, p600 = policies["300"], policies["600"]
    threshold = lambda p: f"{p['threshold']:.2%}" if p["threshold"] is not None else "표시 없음"
    first = cur["equipment"][0]
    op = r["processing"]["fitted_by_role"]["operation"]
    apps = r["processing"]["applications"]
    train_app = apps["operation_train"]
    comparisons = sorted(r["comparisons"], key=lambda x: (-x["ap"], -x["latest_top20_hits"], x["brier"], x["key"]))
    comparison = table(["모델·피처 구성(한글)", "앞 검증 AP", "ROC-AUC", "Brier", "월 20대 적중", "선택"], [
        [x["label"], f"{x['ap']:.4f}", f"{x['roc_auc']:.4f}", f"{x['brier']:.4f}", x["latest_top20_hits"],
         "선택" if x["key"] == r["selected_model"] else ""] for x in comparisons])
    data_table = table(["파일", "정비 기록", "고유 드론", "완전 중복", "드론·날짜 중복"], [
        [name, x["rows"], x["unique_equipment"], x["exact_duplicate_extras"], x["equipment_date_duplicate_extras"]]
        for name, x in profile["files"].items()])
    role_names = {"development": "모델·기준 선택", "evaluation": "하반기 추가 평가",
                  "submission": "2025년 제출", "operation": "2026년 1월 운영"}
    fitted_table = table(["역할", "학습 Q1", "학습 Q3", "하단 울타리", "상단 울타리", "100배 오류 후보 초과 기준"], [
        [role_names[name], x["q1"], x["q3"], x["lower_fence"], x["upper_fence"], x["hundredfold_above"]]
        for name, x in r["processing"]["fitted_by_role"].items()])
    cleaning_table = table(["적용 구간", "행 수", "999시간→N/A", "비용 후보", "100배 가정 포함", "사분위 제한만"], [
        [name, x["rows"], sum(x["time_999_to_na"].values()), x["cost_outliers"],
         x["cost_hundredfold_assumed"], x["cost_quartile_only"]] for name, x in apps.items()])
    split_labels = {"development_train": "모델 선택용", "evaluation_train": "하반기 평가용",
                    "submission_train": "시험 제출용", "operation_train": "월 운영용"}
    split_table = table(["학습 역할", "학습 건", "마지막 정비일", "마지막 정답 관측 종료", "예측 시작"], [
        [split_labels[name], x["rows"], x["end"], x["latest_label_end"], x["prediction_start"]] for name, x in r["splits"].items()])
    monthly_table = table(["월", "최신 후보", "후보 재고장", "모델 적중", "연수 기준", "무작위 1000회 평균"], [
        [x["month"], x["candidates"], x["positives"], x["model_hits"], x["age_hits"], f"{x['random_hits_mean']:.3f}"] for x in latest])
    def policy_table(items):
        return table(["가정(만원)", "기준", "TP", "FP", "FN", "TN", "Precision", "Recall / TPR", "FPR", "비용(만원)"], [
            [key, threshold(policies[key]), x["true_positives"], x["false_positives"], x["false_negatives"], x["true_negatives"],
             f"{x['precision']:.4f}", f"{x['recall']:.4f}", f"{x['fpr']:.4f}", f"{x['cost']:,}"] for key, x in items.items()])
    example_table = table(["특성", "관측값", "학습 참조값", "원 확률", "치환 후 확률", "차이(%p)"], [
        [x["label"], x["value"], x["reference"], f"{x['original_probability']:.2%}",
         f"{x['replaced_probability']:.2%}", f"{x['delta_probability']*100:.3f}"] for x in a["example_factor_recomputed"]])
    pi_table = table(["특성(한글)", "AP 감소 평균", "반복 표준편차"], [
        [x["label"], f"{x['ap_decrease_mean']:.5f}", f"{x['ap_decrease_std']:.5f}"] for x in r["permutation_importance"][:8]])
    waiting = ", ".join(x["equipment_id"] for x in cur["equipment"] if x["risk_flag"] and not x["selected"]) or "없음"
    pattern_rows = []
    import csv
    from pathlib import Path
    with (Path(__file__).resolve().parents[1] / "reports/eda_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        eda = list(csv.DictReader(stream))
    for field in ("드론 유형", "정비 월", "고장 유형"):
        group = [x for x in eda if x["기준"] == field]
        for direction, x in [("높은 범주", max(group, key=lambda x: float(x["재고장률"]))),
                             ("낮은 범주", min(group, key=lambda x: float(x["재고장률"])))]:
            pattern_rows.append([field, direction, x["구분"], x["재고장 건수"], x["정비 건수"], f"{float(x['재고장률']):.2%}"])

    report = f"""# 군 특화 AI 활용 데이터분석 과정 — 드론 예측정비 재분석 결과보고

작성일 2026-11-17 / 분석 대상월 **2026년 1월** / 교육용 가상 자료

[5분 지휘관 발표 대본](2026-01_지휘관발표스크립트.md) · [계산식·출처 대조표](계산식_출처_대조표.md) · [독립 검산](briefing_evidence.json)

지휘관 보고 요지: 최신 드론 **{cur['candidates']}대 중 {cur['selected']}대**를 재점검 대상으로 선정했다.
기본 위험 기준 **{threshold(p300)}** 이상은 **{cur['risk_flagged']}대**, 정원 밖 위험은 **{cur['risk_outside_capacity']}대**다.
선택 구성은 **{r['selected_model_label']}**이다. 별도 하반기 **AP {metric['ap']:.4f} / ROC-AUC {metric['roc_auc']:.4f}**,
월 20대 적중 합 **{total('model_hits')}건**, 연수 기준 **{total('age_hits')}건**, 무작위 평균 합 **{total('random_hits_mean'):.3f}건**이다.
이는 과거 가상 자료의 선정 성능이며 실제 예방 건수·절감액을 측정한 결과는 아니다.

## Ⅰ. AI 활용

### 1. 활용 목적과 사람·AI의 역할

사람은 분석 목적, 30일 정답, 월 20대, 비용 300/600·60만원과 이번 재분석의 999/N/A·비용 오류 가정을 지정했다.
Codex는 README·스키마·분포 확인, 처리 코드 작성, 시간 분할 모델 비교, 영향 요인 계산, 앱·보고서와 검증 코드 작성에 활용했다.
실제 숫자는 Python·공식 머신러닝 라이브러리 실행 결과로 계산했다. 숨겨진 시험 정답을 생성하거나 언어모델의 답변을 정답으로 사용하지 않았다.
최종 데이터 가정·정비 우선순위·운용 판단은 사람이 확인한다.

### 2. AI에 제시한 작업 조건과 실행 흐름

이번 요청을 실행 조건으로 정리하면 “999시간·공란은 N/A, 비용 이상치는 100배 오류 가정과 Q1~Q3 치환,
시간으로 학습·검증 분리, PR-AUC와 월 20대 적중·두 기준선 비교, 검증 비용으로 300/600만원 기준 결정,
한글 모델·피처 구성과 영어 평가 용어로 보고”다. 위 문장은 이번 사용자 요청을 실행 가능한 조건으로 정리한 것이다.

진행은 원본 해시 확인 → 처리 규칙 명문화 → 학습 구간만으로 통계량 추정 → 12개 구성 비교 → 별도 하반기 평가 →
시험 전량 예측·월 20대 선정 → 개별 설명·독립 검산 → 앱·보고서·Git 순서다.
오류는 해당 단계의 입력과 실제 예외를 확인해 수정하며, 진행·검증 기록은 [작업 기록](preparation_log.md)에 남겼다.

### 3. AI 활용의 검증과 한계

공식 방법 정의, 과제 가정, 관측 결과, 프로젝트 판단을 구분했다. 독립 수식으로 AP·PR 면적·ROC·Brier를 재계산했고,
비용표 {a['cost_rows_recomputed']:,}행, 월별 두 집계, 순열 중요도, 개별 참조값 치환과 보정곡선 집계를 대조했다.
새 전처리는 과거 q99 처리를 대체한다. 999와 100배 오류는 **사용자가 지정한 가정**이며 실제 입력 오류가 확인된 사실은 아니다.
AI가 그럴듯한 근거를 제시하더라도 공식 문서·실제 코드·실행 산출물로 확인해야 한다.

## Ⅱ. 데이터 분석

### 1. 데이터 품질·분포·중복

{data_table}

드론 ID 반복은 정상 정비 이력이다. 이번 달 106건은 최신 81대로 정리하되 이전 25행도 이력 계산용으로 보존한다.
원본 삭제는 0행이다. 학습 양성은 525/2346이며 시험·이번 달 정답은 제공되지 않았다.

{table(['집계','방향','범주','재고장','정비 건','재고장률'], pattern_rows)}

위 분포는 학습 전체의 기술 통계다. 반복 드론·연식·정비 구분 등이 섞여 있어 유형·계절·고장 종류의 인과효과로 단정하지 않는다.
근거: data_profile.json, eda_summary.csv, F01.

### 2. N/A·비용 치환 기준과 실제 처리량

운용중단/수리 시간의 **999와 숫자 공란은 N/A**로 취급한다. 모델은 학습 중앙값으로 대체하고 결측 지시열을 남긴다.
고장 코드·유형 공란 등 범주 공란도 N/A다. 학습 고장 코드·유형 공란은 각 1,707건으로 모두 예방정비 기록에 있으며,
직전 정비 경과일 공란 150건은 첫 관측 정비다. N/A의 발생 원인이 확인된 것과는 구분한다.

비용은 학습 Q1·Q3로 IQR=Q3−Q1과 울타리 Q1−1.5IQR, Q3+1.5IQR을 계산한다.
울타리 밖 후보 중 **100×Q3 초과**는 100배 입력 오류로 가정하여 100으로 나누고 Q1~Q3로 제한한다.
나머지 후보는 원 값을 Q1~Q3로 제한한다. 울타리 안의 유효값은 유지하므로 모든 비용을 중앙 50%로 몰아넣은 처리는 아니다.
각 수치열의 학습 중앙값은 해당 열의 Q1~Q3 안에 있다. IQR 후보 탐색은 {src('S19')}, 선형 분위수 계산은 {src('S08')}·{src('S09')}를 참고했다.
**100배 가정·100×Q3 후보 기준·치환 범위는 프로젝트 규칙**이며 NIST나 군 원가 표준이 정한 오류 보정법이 아니다.

{fitted_table}

{cleaning_table}

운영 학습에서는 999시간 {sum(train_app['time_999_to_na'].values())}건을 N/A로 바꿨다.
비용 후보 {train_app['cost_outliers']}건 중 100배 가정 {train_app['cost_hundredfold_assumed']}건,
사분위 제한만 적용한 값 {train_app['cost_quartile_only']}건이다. 19,000만원은 190만원으로 나눈 후
운영 Q3={op['q3']:g}만원으로 제한한다. 이는 실제 비용이 {op['q3']:g}만원이었다는 확인이 아니다.
시험에는 제출 모델의 Q1~Q3를, 이번 달에는 운영 모델의 Q1~Q3를 적용한다.
시험·이번 달 분포로 통계량을 다시 맞추지 않는다. 행 삭제·원본 CSV 수정은 없다.
전후 값·역할·원본 행·모델 대체값: [치환 감사 CSV](cleaning_audit.csv), 산식 F02·F03.

### 3. 피처 구성과 제외 이유

{table(['피처 묶음(한글)','내용'], [[FEATURE_GROUPS['basic'],'현재 정비 정보·상태·월 주기'], [FEATURE_GROUPS['history'],'기본 묶음에 현재일 이전 정비·고장·예방정비 이력 추가'], [FEATURE_GROUPS['context'],'이력 묶음에 부대와 고장 코드 함께 추가']])}

최종 입력은 **{len(r['processing']['feature_columns'])}개**다:
{', '.join(r['processing']['feature_labels'])}.

이력은 현재 정비일보다 앞선 동일 드론 기록만 사용한다. 당일 다른 정비와 미래 기록·미래 정답은 제외한다.
90일 정비·사후/긴급정비 수, 고장 후 경과일, 예방정비 이후 비행시간 증가, 180일 실패 수 등을 계산했다(F04·F05).
90/180일은 프로젝트 설계이며 원래 군 표준 창이라는 근거는 없다.

{table(['항목','사용·제외 이유'], [['드론 번호','이력 결합·제출 식별에 사용, 모델 ID 암기 입력은 제외'], ['원 정비일','시간 분할·월 주기·이력에 사용, 날짜 직접 입력은 제외'], ['월 가동률','정비 직후 월말값의 가용성 미확인·이번 달 열 부재'], ['재고장 정답','예측할 대상이며 입력에서 제외'], ['부대·고장 코드','묶음 추가 구성을 모든 알고리즘에서 비교; 최종 포함 여부는 선택된 구성에 따름']])}

부대·고장 코드 각각의 독립 효과는 분리 실험하지 않았다. 화면·보고서의 피처 이름은 한글이며,
재현용 JSON·CSV의 내부 컬럼명과 모델 키는 영문으로 유지했다.

### 4. 시간 분할·30일 정답 성숙

{split_table}

앞 검증 2024-01~05의 {r['development']['metrics']['rows']}건으로 모델·위험 기준을 선택하고,
뒤 검증 2024-07~12의 {metric['rows']}건은 선택된 절차의 추가 평가에 사용했다.
학습은 정비일+30일 < 예측 시작일인 행만 허용한다(F06). 6월 결과를 보고 7월 모델을 선택하지 않았다.
2025 시험과 2026년 1월의 정답은 쓰거나 복원하지 않았다.

### 5. 12개 모델·피처 재비교와 선정 이유

모든 구성에 동일한 새 N/A·비용 전처리를 적용했다. 4개 알고리즘 × 3개 피처 묶음을 앞 검증에서 비교했다.
모델 설정은 실행 전 고정했으며 전체 최적 튜닝·유의성 검정은 수행하지 않았다.

{comparison}

선택 규칙은 AP 내림차순 → 동률이면 월 20대 적중 내림차순 → Brier 오름차순 → 내부 키 순이다.
선택 구성 **{r['selected_model_label']}**, 앞 AP **{comparisons[0]['ap']:.4f}**다.
월 20대 적중이 더 큰 다른 구성이 있을 수 있으나, 사전에 정한 AP 우선 규칙을 결과를 보고 바꾸지 않았다.
한글 피처 묶음만 다르게 하여 알고리즘마다 같은 3개 구성을 비교했다. 뒤 검증으로 재선정하지 않았다.

선택된 랜덤 포레스트는 깊이 6·최소잎 8·240개 트리·sqrt 피처 표본·bootstrap·seed 42의 고정 설정이다.
확률은 각 트리 잎의 재고장 비율을 평균한다. 트리 평균 가중치는 1/240이며 사람이 정한 요인별 점수 가중치는 없다(F07).
이 결과로 모든 기종·부대·다른 튜닝에서도 해당 모델이 우수하다고 결론 내리지는 않는다.

### 6. PR-AUC·ROC-AUC·영어 평가 용어

{table(['지표','계산/해석','뒤 검증 값'], [['PR-AUC 주 지표 AP','Σ(Recall 증가량×해당 Precision)',f"{metric['ap']:.6f}"], ['사다리꼴 PR-AUC','인접 PR 점의 선형 면적; AP와 구분',f"{metric['pr_auc_trapezoid']:.6f}"], ['ROC-AUC','양성과 음성 확률의 순위 구분력; 동점 0.5',f"{metric['roc_auc']:.6f}"], ['Brier','평균 (예측확률−정답)²; 낮을수록 작음',f"{metric['brier']:.6f}"]])}

Precision=TP/(TP+FP)는 표시한 사례 중 실제 재고장의 비율, Recall=TP/(TP+FN)은 실제 재고장 중 표시한 비율이다.
True Positive Rate (TPR)는 이 이진 평가에서 Recall과 같고, False Positive Rate (FPR)=FP/(FP+TN)는 비재고장 중 잘못 표시한 비율이다(F08).
PR 지표는 제한된 자원에서 재고장을 높은 순위에 모으는 목적에 맞춰 주 지표로 사용했다.
ROC-AUC는 전체 양성·음성 순위 구분의 보조 지표로, 월 20대 적중과 함께 해석한다. AP는 정확도가 아니다.
Brier와 분위수 5개 구간 점검(F17)만으로 확률 보정이 완료됐다고 판단하지 않는다. 출처: {src('S03')}·{src('S05')}·{src('S20')}.

### 7. 월 20대 적중과 두 기준선

{monthly_table}

최신 고유 드론 후보 합 {total('candidates')}드론-월, 후보 양성 {total('positives')}건, 선정 {total('selected')}건이다.
모델 적중 {total('model_hits')}건 / 연수 기준 {total('age_hits')}건 / 무작위 1000회 평균 합 {total('random_hits_mean'):.3f}건이다.
같은 드론이 다른 월에 재선정되면 별개 사건이다. 무작위 기대값은 K×M/N, 반복은 seed 42와 월별 파생 시드로 고정했다(F12·F13).
무작위 결과 분위수는 평균의 신뢰구간이 아니다.
정비 기록 상위 20건의 적중 합 {sum(x['model_hits'] for x in records)}건은 중복 드론을 포함할 수 있는 별도 집계다.
AP의 {metric['rows']}건, 최신 {total('candidates')}드론-월, 선정 {total('selected')}건, 이번 달 {cur['candidates']}대를 구분한다.

### 8. 검증 비용으로 위험 기준 선택

비용=FN×미탐지비용+(TP+FP)×60만원. 단가 300/600·60만원은 README의 교육용 가정이다.
앞 검증에서 가능한 판정 변경점과 모두 표시·표시 없음 정책을 비교했다. 동률이면 표시 수가 적고 기준이 높은 정책을 선택했다.
판정은 p≥기준이다(F14). 2025 시험으로 기준을 고르지 않았다.

{policy_table(policies)}

실제 대입: {p300['false_negatives']}×300+{p300['flagged']}×60={p300['cost']:,}만원,
{p600['false_negatives']}×600+{p600['flagged']}×60={p600['cost']:,}만원이다.
미탐지 가정을 300에서 600만원으로 바꾸면 기준은 **{threshold(p300)} → {threshold(p600)}**다.

앞 기준을 뒤 검증에 고정 적용한 결과:

{policy_table(r['evaluation']['frozen_policy_costs'])}

위 비용은 정원 제한 없는 정비 기록별 위험 표시 비용이다. 월 20대 운영 비용은 같은 최신 후보에서
(후보양성−선정적중)×300+선정대수×60으로 별도 계산한다. 실제 예방·절감 효과를 측정한 것이 아니다.
완전 보정·완전 예방 가정의 60/300=20%, 60/600=10%는 이론 참고식이며 실제 사용 기준은 검증 최소비용 기준이다.

### 9. 영향 요인·이번 달 계획·앱 시연

첫 드론 **{first['equipment_id']}**, {first['equipment_type']}·{first['unit_code']}, {first['maintenance_date']}·{first['maintenance_type']},
모델 확률 **{first['prob']:.2%}**:

{example_table}

한 입력을 학습 참조값으로 바꾼 모델 확률 차이다(F15). 인과·실제 예방 효과·SHAP·합산 가중치로 해석하지 않는다.
원본에 999가 있어도 모델은 N/A와 학습 중앙값으로 처리하며, 감사 CSV에서 원본과 모델 입력을 구분할 수 있다.
전체 검증 영향 요인은 특성을 5회 섞었을 때의 AP 감소다. 개별 민감도와 모델·표본·척도가 다르다.

{pi_table}

정원 밖 위험 {cur['risk_outside_capacity']}대: **{waiting}**.
정비반장이 확률 순 대기·운용 전 상태 확인·추가 점검 가능성을 검토한다.
전체 20대와 권고는 [재점검 계획](reinspection_plan_2026-01.md)에 있다.
앱은 재점검 계획 상단 → 데이터·과정의 처리 기준 → 모델 검증의 한글 비교표·곡선·비용 → 정렬·유형/정비 요약 상세 →
돌아가기 → 첫 드론 설명 순으로 시연한다. 필터·보기 정렬은 전체 선정 순위를 바꾸지 않는다.

### 10. 근거·재현·한계

숫자: analysis_results.json / 처리 전후: cleaning_audit.csv / 수식·함수·공식 출처: 계산식 대조표 /
독립 검산: briefing_evidence.json / 해시: output_manifest.json·briefing_manifest.json / 진행: preparation_log.md.
실행 순서는 python scripts/train.py → python scripts/build_report.py → python scripts/build_briefing.py →
python scripts/check.py --predictions reports/test_predictions_2025.csv 이다.

가상 데이터이며 실제 비용 오류·예방 효과·다른 기종/부대의 일반화를 검증하지 않았다.
비용 이상치에는 실제 고비용 정비가 섞일 수 있어 사분위 치환의 정보 손실이 있다.
2025 시험·이번 달 정답이 없으므로 실제 성능은 강사 채점 전까지 알 수 없다.
재학습 후 확률 척도·고정 기준을 다시 검증해야 한다. 이번 달은 월말 최신 기록 계획이다.
**이 하반기 자료는 이전 분석에서도 이미 확인한 기간이므로, 이번 수정까지 포함한 완전히 새로운 외부 검증으로 표현하지 않는다.**
후속 미사용 기간·다른 부대 검증과 실제 개입 효과 확인이 필요하며 최종 판단은 담당자가 한다.
"""

    formulas = [
        ("F01", "건수·중복·관측 비율", r"D_{exact}=n-|\{row_i\}|,\quad D_{key}=n-|\{(e_i,t_i)\}|,\quad \hat\pi=\sum_i y_i/n",
         "드론 반복은 이력이며 완전 중복과 다르다. 정답은 제공 학습 CSV에서만 읽는다.", "scripts/profile_data.py", "profile_file", "data_profile.json; eda_summary.csv", "제공 자료의 직접 집계"),
        ("F02", "사분위·IQR 후보·비용 치환", r"h=(n-1)q,\ j=\lfloor h\rfloor,\ a=h-j,\ Q_q=(1-a)x_{(j)}+ax_{(\min(j+1,n-1))}",
         r"0부터 인덱스의 선형 분위수다. $L=Q1-1.5(Q3-Q1), U=Q3+1.5(Q3-Q1)$, $O=1[c<L\lor c>U]$. $r=c/100$ if $O=1,c>100Q3$; 그 외 $r=c$. $c'=\min(Q3,\max(Q1,r))$ if $O=1$; 그 외 $c'=c$. 수치 공란은 유지하여 F03으로 넘긴다. 이 규칙은 사용자 오류 가정을 구체화한 설계이며 오류 확정이 아니다.",
         "rdm_model.py", "QuartileRepairer", "processing.fitted_by_role / applications; cleaning_audit.csv", src("S08")+"; "+src("S09")+"; "+src("S19")+" (NIST는 울타리 정의만 지원)"),
        ("F03", "N/A·학습 중앙값·결측 표시·표준화", r"x^*=\operatorname{median}(X^{train})\ \text{if missing},\quad b=1[\text{missing}],\quad z=(x^*-\mu^{train})/\sigma^{train}",
         r"운용중단/수리 999→NaN, 공란→N/A. 수치 중앙값·평균·표준편차는 학습에서만 추정한다. $\sigma=\sqrt{\sum_i(x_i^*-\mu)^2/n}$, 상수 scale=1. 범주 N/A는 별도 범주, 원핫 $z_c=1[x=c]$, 새로운 범주는 해당 블록 0이다. 각 수치열의 중앙값은 해당 열의 Q1~Q3 안이다. 시간은 시간 단위, 비용은 만원 단위로 각각 대체한다.",
         "rdm_model.py", "make_model", "cleaning_audit.csv: cleaned / model_imputed; saved model prep", src("S10")+"; "+src("S11")+"; "+src("S18")+"; 999 규칙은 사용자 지정"),
        ("F04", "현재일 이전 과거 이력", r"H_e(t)=\{j:e_j=e,t_j<t\},\quad N_{90}=\sum_{j\in H_e(t)}1[t-90d\le t_j<t]",
         r"사후·긴급정비 수는 해당 조건의 합, 최근 고장 경과일은 $t-\max t_{fault}$, 예방정비 후 비행시간은 $\max(0,h_t-h_{prevPM})$, 180일 실패 수는 창 내 success=0 합이다. 횟수 부재는 0, 경과일/시간 부재는 NaN·존재 표시는 0이다.",
         "rdm_model.py", "build_features", "processing.feature_labels", "README 과제를 구체화한 프로젝트 설계; 90/180일은 표준 규정 아님"),
        ("F05", "월 주기", r"s_m=\sin(2\pi m/12),\quad c_m=\cos(2\pi m/12)",
         "12월·1월의 인접성을 표현한 달력 변환이다.", "rdm_model.py", "build_features", "processing.feature_labels", "달력 12개월 주기의 수학적 정의"),
        ("F06", "30일 정답 성숙 경계", r"\mathcal T(T)=\{i:source_i=train,\ t_i+30d<T\}",
         "행 간 gap=30을 쓴 것이 아니라 날짜로 30일 정답 관측 종료를 확인한다.", "scripts/train.py", "training_mask", "splits", src("S12")+" 개념 참고; 구현은 직접 날짜 마스크"),
        ("F07", "랜덤 포레스트 확률·학습 가중치", r"\hat p(x)=\frac1{240}\sum_{b=1}^{240}q_b(x),\quad q_b(x)=\frac{\sum_{i\in leaf_b(x)}w_{bi}y_i}{\sum_{i\in leaf_b(x)}w_{bi}}",
         r"부트스트랩 중복 표본 가중치를 반영한 잎 비율이다. $G(t)=1-\sum_k q_{tk}^2$, $\Delta G=G(t)-(W_L/W_t)G(L)-(W_R/W_t)G(R)$. 평균 가중치 1/240은 트리 간 가중치이며 요인별 수동 점수가 아니다.",
         "rdm_model.py", "make_model", "selected_spec; briefing_evidence.json: random_forest_parameters", src("S01")+"; "+src("S02")),
        ("F08", "혼동행렬·Precision·Recall·TPR·FPR", r"TP=\sum y_i a_i,\ FP=\sum(1-y_i)a_i,\ FN=\sum y_i(1-a_i),\ TN=\sum(1-y_i)(1-a_i),\ a_i=1[p_i\ge\tau]",
         r"$Precision=TP/(TP+FP)$, $Recall=TP/(TP+FN)$, $TPR=Recall$, $FPR=FP/(FP+TN)$. 0분모는 None이다.",
         "rdm_model.py", "evaluate_policy", "threshold_policies; evaluation.frozen_policy_costs", src("S20")+"; "+src("S03")+"; "+src("S05")),
        ("F09", "AP와 사다리꼴 PR 면적", r"AP=\sum_g(R_g-R_{g-1})P_g,\quad PR\!-\!AUC_{trap}=\sum_g(R_g-R_{g-1})(P_g+P_{g-1})/2",
         "동점 점수 그룹을 함께 포함하고 Recall 증가 방향으로 계산한다. 시작 R0=0,P0=1. 양성·음성 각 1건이 같은 점수면 AP=.5, 사다리꼴=.75다. P는 Precision, R은 Recall이다.",
         "rdm_model.py", "metrics", "evaluation.metrics; briefing_evidence.json: manual_holdout_metrics", src("S03")+"; "+src("S04")),
        ("F10", "ROC-AUC의 순위 검산", r"AUC=\frac{\sum_{i:y_i=1}\sum_{j:y_j=0}(1[p_i>p_j]+\frac12 1[p_i=p_j])}{n_+n_-}",
         "양성·음성 쌍을 직접 비교하고 동점은 0.5로 센다. 정확도가 아니다.", "scripts/build_briefing.py", "manual_metrics", "manual_holdout_metrics.roc_auc", src("S05")+"; 쌍 비교는 동등한 이진 순위 검산식"),
        ("F11", "Brier 확률 오차", r"Brier=n^{-1}\sum_i(p_i-y_i)^2",
         "이진 0~1 척도, 낮을수록 작다. 단독으로 확률 보정 완료를 입증하지 않는다.", "rdm_model.py", "metrics", "evaluation.metrics.brier", src("S06")),
        ("F12", "최신 드론 월 20대 적중", r"i^*_{e,m}=\operatorname{argmax}_{i:e_i=e,month(t_i)=m}t_i,\quad S_m=TopK_p\{i^*_{e,m}\},\quad H_m=\sum_{i\in S_m}y_i",
         "K=min(20,후보수), 확률 동점은 드론 번호순이다. 최신 날짜 충돌은 임의 결정하지 않는다. 다른 월의 같은 드론은 별개 선정이다.", "rdm_model.py", "monthly_topk", "evaluation.monthly; monthly_validation.csv", "README 정원·최신 기록 규칙을 구체화한 프로젝트 정의"),
        ("F13", "무작위 기대값·반복 평균", r"E[H_m]=K_m M_m/N_m,\quad \bar H_m=1000^{-1}\sum_{r=1}^{1000}H_{m,r}",
         "같은 월·후보에서 중복 없이 선택, seed=42+YYYYMM+(정비 기록이면 1). 분위수는 무작위 선정 결과 분포이며 평균 신뢰구간이 아니다.", "rdm_model.py", "monthly_topk", "random_expected_hits / random_hits_mean", "후보 선정확률 K/N의 합으로 유도; 반복수·시드는 프로젝트 설정"),
        ("F14", "300/600만원 비용·위험 기준", r"C_c(\tau)=cFN(\tau)+60(TP(\tau)+FP(\tau)),\quad \tau_c^*=\operatorname{argmin}_\tau C_c(\tau),\quad c\in\{300,600\}",
         r"앞 검증에서 선택, 뒤 검증에는 고정한다. 동률은 표시 수 최소·기준 최대 순이다. 월 정원 비용 $c(M-H)+60K$는 별도다. 완전 보정·완전 예방 시 이론 기대손실 분기점 60/c는 실제 경험 최적 기준과 다를 수 있다.",
         "rdm_model.py", "cost_sweep", "threshold_costs.csv; threshold_policies", "README 비용 가정; 경험 최적화·동률 규칙은 프로젝트 설계"),
        ("F15", "개별 민감도·전체 순열 중요도", r"\Delta_j=\hat p(x)-\hat p(x_{j\leftarrow r_j}),\quad \Delta_j^{pp}=100\Delta_j,\quad I_j=s_0-\frac15\sum_{r=1}^5s_{j,r}",
         r"개별 참조값은 운영 학습 중앙값/최빈값, 전체 점수는 하반기 AP다. 표준편차는 $\sqrt{5^{-1}\sum_r[(s_0-s_{j,r})-I_j]^2}$다. 서로 합산하지 않으며 SHAP·인과·예방 효과가 아니다.",
         "scripts/train.py", "explain_reference", "current.equipment.factors; permutation_importance", src("S07")+"; 개별 치환은 프로젝트 설명 방식"),
        ("F16", "비교 모델 출력·설정", r"p_{LR}=\sigma(\beta_0+\beta^Tz),\quad p_{boost}=\sigma(F_0+\eta\sum_m f_m),\quad \sigma(a)=(1+e^{-a})^{-1}",
         "로지스틱 C=.5,max_iter=2000; 히스토그램 부스팅 학습률 .04,160회,잎15,L2=2; 라이트지비엠 학습률 .035,180개,잎15,최소자식20,L2=2. 12개 구성 모두 동일 전처리이며 최적 튜닝 실험은 아니다.",
         "rdm_model.py", "make_model", "comparisons; selected_spec", src("S13")+"; "+src("S14")+"; "+src("S15")+"; "+src("S16")),
        ("F17", "확률 구간 점검", r"\bar p_b=n_b^{-1}\sum_{i\in B_b}p_i,\quad \bar y_b=n_b^{-1}\sum_{i\in B_b}y_i",
         "확률 분위수 5개 구간의 평균 예측과 관측 비율이다. 보정기를 별도 학습하지 않았다.", "scripts/train.py", "main", "evaluation.calibration", src("S17")),
    ]
    appendix = "# 드론 예측정비 계산식·출처·구현 대조표\n\n작성일 2026-11-17 / 새 N/A·비용 사분위 처리 재분석\n\n"
    appendix += "e=드론, t=정비일, y=제공된 재고장 정답, p=모델 확률, K=정원이다. 시험·이번 달 y는 제공되지 않았다. 1[조건]은 표시함수다.\n\n"
    appendix += table(["식","계산","구현"], [[x[0],x[1],code(x[4],x[5])] for x in formulas]) + "\n\n"
    for key,title,equation,meaning,path,symbol,fields,source in formulas:
        appendix += f"## {key}. {title}\n\n$$\n{equation}\n$$\n\n{meaning}\n\n- 구현: {code(path,symbol)}\n- 결과: {fields}\n- 출처·적용: {source}\n\n"
    appendix += "## 실제 비용 대입과 독립 검산\n\n"
    for cost,p in policies.items():
        appendix += f"- {cost}만원 가정: {p['false_negatives']}×{cost}+{p['flagged']}×60={p['cost']:,}만원, 기준 {threshold(p)}.\n"
    appendix += f"\nAP·PR 면적·ROC·Brier를 별도 수식으로 검산하고 원 실행과 1e-12 이내 일치했다. 비용표 {a['cost_rows_recomputed']}행과 월별 기준선·개별 설명·순열 중요도도 대조했다. 검산 과정에서는 저장 모델·원본을 바꾸지 않았다.\n\n## 공식 출처\n\n"
    appendix += "\n".join("- "+src(f"S{i:02}") for i in range(1,21))
    appendix += "\n\n공식 정의와 프로젝트 가정을 구분한다. NIST의 IQR 울타리 설명과 NumPy/pandas의 선형 분위수 알고리즘을 사용했으며, NIST 예제의 별도 분위수 위치식으로 구현을 바꾸지는 않았다. 비용·30일·20대는 README, 999·100배 오류는 사용자 지정 가정, 후보 세분화·치환·분할·시드·피처 창은 프로젝트 설계다.\n"

    speech = f"""# 드론 예측정비 재분석 · 지휘관 5분 발표 대본

교육용 가상 자료 / 2026년 1월 / [분석결과보고](2026-01_분석결과보고.md) · [계산식·출처](계산식_출처_대조표.md)

## 발표 전 확인

권장 진입점 app_styled.py, 앱 http://127.0.0.1:8517 에서 재점검 가능 20대, 임계값 직접 조정 끔, 필터·검색 없음, 목록 선정 대상으로 맞춘다. 기존 배포 app.py도 같은 rdm_ui 화면을 사용한다.
상단 {cur['selected']} / {cur['candidates']} / {cur['risk_flagged']} / {cur['risk_outside_capacity']}, 기본 기준 {threshold(p300)},
첫 드론 {first['equipment_id']} ({first['prob']:.2%})를 확인한다.
자료 생성·검사 명령: python scripts/build_briefing.py, python scripts/check.py --predictions reports/test_predictions_2025.csv.
발표 대사만 낭독하고 표·질의는 참고한다. 조작 응답 시간을 포함해 5분으로 리허설한다.

## 00:00–00:35 · 결론

화면: 재점검 계획 상단

**발표 대사**

“지휘관님, 드론 예측정비 재분석 결과를 보고드리겠습니다. 제공된 2026년 1월 최신 드론 {cur['candidates']}대 중 {cur['selected']}대를 선정했습니다. 위험 기준 이상은 {cur['risk_flagged']}대, 정원 밖 위험은 {cur['risk_outside_capacity']}대입니다. 보고는 AI 활용과 데이터 분석으로 구분하겠습니다.”

## 00:35–01:15 · AI 활용

화면: 데이터·과정의 분석 진행 순서

**발표 대사**

“사람이 정한 과제와 처리 가정을 바탕으로 Codex를 코드 작성, 모델 비교, 앱과 보고서, 검증 코드 작성에 활용했습니다. 수치는 Python 실행 결과에서 가져왔고 공식 수식과 별도 검산으로 확인했습니다. AI가 시험 정답이나 실제 예방 효과를 만들어낸 것은 아닙니다. 데이터 가정과 최종 정비 판단은 사람이 확인합니다.”

## 01:15–02:05 · 데이터 처리

화면: 데이터·과정의 새 처리 기준과 한글 피처 목록

**발표 대사**

“999시간과 공란은 N/A로 처리했습니다. 숫자는 학습 중앙값과 결측 표시를 사용합니다. 비용은 학습 사분위수로 이상치 후보를 찾고, 매우 큰 값에는 100배 입력 오류를 가정한 뒤 Q1부터 Q3로 제한했습니다. 운영 학습에서 999시간 {sum(train_app['time_999_to_na'].values())}건, 비용 후보 {train_app['cost_outliers']}건을 처리했습니다. 원본과 행은 보존했습니다. 이는 확인된 입력 오류가 아닌 분석 가정이며 실제 고비용 정보가 줄 수 있습니다.”

## 02:05–03:05 · 모델·지표·기준선

화면: 모델 검증의 한글 비교표와 PR·ROC 곡선·월별 기준선

**발표 대사**

“같은 전처리로 4개 알고리즘과 3개 피처 묶음, 총 12개를 비교해 {r['selected_model_label']}을 선택했습니다. 앞 검증에서 고르고 뒤 하반기로 평가했습니다. 하반기 AP는 {metric['ap']:.4f}, ROC-AUC는 {metric['roc_auc']:.4f}입니다. AP는 Precision과 Recall을 요약하며 정확도가 아닙니다. 월 20대 적중은 모델 {total('model_hits')}건, 연수 기준 {total('age_hits')}건, 무작위 평균 {total('random_hits_mean'):.1f}건입니다. 시험 2025년 성능은 정답이 없어 보고하지 않습니다.”

## 03:05–04:00 · 선정·영향 요인 시연

화면: 재점검 계획에서 부대순 정렬 → 확률순 복원 → 감시드론 이름 클릭 → 돌아가기 → 첫 드론 상세

**발표 대사**

“드론, 유형, 부대, 최근 정비일, 정비 구분, 확률로 정렬할 수 있습니다. 요약 이름을 누르면 해당 전체 후보가 나오고 돌아가면 원래 목록이 복원됩니다. 보기 정렬과 전체 선정 순위는 별개입니다. 첫 드론 {first['equipment_id']}의 확률은 {first['prob']*100:.2f}퍼센트입니다. 한 입력을 학습 참조값으로 바꾼 확률 차이와 과거 이력을 함께 확인합니다. 요인별 수동 가중치나 실제 고장 원인·예방 효과를 뜻하지 않습니다.”

## 04:00–05:00 · 비용 기준·한계·검토

화면: 모델 검증 비용 표와 사이드바

**발표 대사**

“앞 검증에서 놓친 재고장 300만원과 표시 건당 60만원의 비용을 최소화한 기준은 {threshold(p300)}입니다. 미탐지 비용 600만원이면 {threshold(p600)}입니다. 정원 20대와 별도 설정이고 시험 자료로 기준을 고르지 않았습니다. 정원 밖 위험 {cur['risk_outside_capacity']}대는 대기와 추가 점검을 검토해야 합니다. 실제 비용·예방 효과와 다른 부대 검증은 남아 있습니다. 하반기는 이전에도 확인한 자료여서 새로운 외부 검증으로 주장하지 않습니다. 최종 정비 판단은 담당자가 하도록 제안드립니다. 이상입니다.”

## 질의응답

"""
    faqs = [
        ("중복은 없었는가?", "완전 중복과 드론·날짜 키 중복은 0이다. 이번 달 106건을 최신 81대로 정리한 것은 정상 반복 이력이다. F01·F12."),
        ("999와 공란은 어떤 N/A인가?", "시간 999·숫자 공란은 NaN으로 두고 학습 중앙값과 결측 표시를 적용한다. 범주 공란은 N/A 범주다. 원본·치환 전후는 감사 CSV에 남는다. F03."),
        ("왜 100배 오류와 Q1~Q3인가?", "사용자 지정 오류 가정이다. 학습 IQR 울타리 밖 중 100×Q3 초과에 1/100을 적용하고 모든 후보를 Q1~Q3로 제한한다. 확인된 오류라고 단정하지 않는다. F02."),
        ("비용을 모두 Q1~Q3로 바꿨는가?", "울타리 밖 후보만 바꾸고 유효값은 유지한다. 학습에서 추정한 범위를 검증·시험에 고정한다. 고비용 정보가 줄 수 있다는 한계가 있다."),
        ("어떤 모델과 피처를 비교했는가?", "로지스틱 회귀·랜덤 포레스트·히스토그램 경사 부스팅·라이트지비엠 각각에 현재 상태, 과거 이력 추가, 부대·고장 코드 추가의 3구성을 비교했다."),
        ("월 20대 적중이 더 큰 구성이면 바꾸는가?", "사전에 정한 AP 우선, 동률일 때 월 20대 적중·Brier·키 규칙을 유지했다. 하반기를 보고 재선정하지 않았다."),
        ("Precision·Recall·TPR·FPR의 뜻은?", "Precision은 표시 중 실제 양성 비율, Recall과 TPR은 전체 양성 중 표시 비율, FPR은 전체 음성 중 잘못 표시한 비율이다. F08."),
        ("AP와 PR-AUC는 같은가?", "주 PR 지표는 AP=ΣRecall 증가량×Precision이다. 사다리꼴 PR 면적은 별도로 표시한다. ROC-AUC는 순위 구분의 보조 지표다. F09·F10."),
        ("임계값의 계산식과 출처는?", f"README 비용 가정으로 FN×300+표시×60을 최소화했다. {p300['false_negatives']}×300+{p300['flagged']}×60={p300['cost']:,}만원이다. 600만원도 같은 방식이다. F14."),
        ("왜 20%가 아니라 경험 최적 기준인가?", "60/300은 완전 보정·완전 예방을 가정한 이론 분기점이다. 실제 기준은 검증 확률·정답의 경험 비용 최소값이며 재학습 후 재검증이 필요하다."),
        ("설명 막대가 가중치인가?", "참조값 치환 확률 민감도다. 전체 순열 중요도는 별도의 AP 감소다. 두 수치를 합산하거나 SHAP·인과·예방 효과로 해석하지 않는다. F15."),
        ("추가 검증과 정원 밖 위험은?", "초과 위험은 안전으로 바꾸지 않고 대기·운용 전 확인·자원 추가를 검토한다. 시험 성능·실제 예방 효과·미사용 기간과 다른 부대 검증은 남아 있다."),
    ]
    for i,(q,answer) in enumerate(faqs,1):
        speech += f"### {i}. {q}\n\n{answer}\n\n"
    speech += "## 시연 복구\n\n정렬은 확률 내림차순, 목록은 선정 대상, 정원 20대·직접 임계값 조정 끔·필터 없음으로 복원한다. 요약 상세에서는 돌아가기를 누른다. 지연이 길면 저장된 재점검 계획과 보고서 표로 전환하고 발표 중 재학습하지 않는다. 정비 구분별 사후정비 상세는 시간 여유가 있을 때만 추가 시연한다.\n\n## 참고 근거\n\n[수식·공식 출처](계산식_출처_대조표.md), [보고서](2026-01_분석결과보고.md), [계획](reinspection_plan_2026-01.md), [진행 기록](preparation_log.md).\n"
    return {"reports/2026-01_분석결과보고.md": report,
            "reports/2026-01_지휘관발표스크립트.md": speech,
            "reports/계산식_출처_대조표.md": appendix}
