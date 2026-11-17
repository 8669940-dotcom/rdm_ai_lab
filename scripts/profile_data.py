"""Read the three original CSVs once; save compact evidence without changing data."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date

from rdm_common import EXPECTED, NUMERIC_COLUMNS, ROOT, TARGET, latest_record_indices, read_csv, sha256


def quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def profile_file(filename: str) -> tuple[dict, list[dict[str, str]]]:
    path = ROOT / "data" / filename
    columns, rows = read_csv(path)
    numeric = {}
    for column in NUMERIC_COLUMNS:
        if column not in columns:
            continue
        numbers, invalid = [], 0
        for row in rows:
            raw = row[column].strip()
            if not raw:
                continue
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError("nonfinite")
                numbers.append(value)
            except ValueError:
                invalid += 1
        numbers.sort()
        if not numbers:
            numeric[column] = {"n": 0, "invalid": invalid}
            continue
        q1, q3 = quantile(numbers, .25), quantile(numbers, .75)
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        numeric[column] = {
            "n": len(numbers), "invalid": invalid, "min": numbers[0],
            "q1": q1, "median": quantile(numbers, .5), "q3": q3, "max": numbers[-1],
            "negative": sum(value < 0 for value in numbers),
            "iqr_flags": (None if column in ("severity", "maintenance_success")
                          else sum(value < low or value > high for value in numbers)),
        }
        if column in ("severity", "availability_rate", "maintenance_success"):
            numeric[column]["domain_invalid"] = sum(
                not (0 <= value <= 5 and value.is_integer()) if column == "severity"
                else value not in (0, 1) if column == "maintenance_success"
                else not 0 <= value <= 1 for value in numbers
            )
        if column == "parts_replaced_count":
            numeric[column]["domain_invalid"] = sum(not v.is_integer() or v < 0 for v in numbers)
    dates, bad_dates = [], []
    for index, row in enumerate(rows):
        try:
            dates.append(date.fromisoformat(row["maintenance_date"]).isoformat())
        except ValueError:
            bad_dates.append(index + 2)  # CSV physical line, including header
    key_counts = Counter((r["equipment_id"], r["maintenance_date"]) for r in rows)
    missing = {c: sum(not r[c].strip() for r in rows) for c in columns}
    grouped = {}
    for maintenance_type in sorted({r["maintenance_type"] for r in rows}):
        subset = [r for r in rows if r["maintenance_type"] == maintenance_type]
        grouped[maintenance_type] = {"rows": len(subset), **{
            c + "_missing": sum(not r[c].strip() for r in subset)
            for c in ("fault_code", "fault_category")}}
    labels = Counter(r[TARGET] for r in rows) if TARGET in columns else None
    largest_examples = {}
    for column in ("downtime_hours", "repair_duration_hours", "maintenance_cost"):
        valid = []
        for index, row in enumerate(rows):
            try:
                value = float(row[column])
                if math.isfinite(value):
                    valid.append((value, index, row))
            except ValueError:
                continue
        valid.sort(key=lambda item: (-item[0], item[1]))
        largest_examples[column] = [
            {"csv_line": index + 2, "equipment_id": row["equipment_id"],
             "maintenance_date": row["maintenance_date"],
             "maintenance_type": row["maintenance_type"], "value": value}
            for value, index, row in valid[:3]
        ]
    profile = {
        "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
        "rows": len(rows), "columns": columns, "column_count": len(columns),
        "date_min": min(dates) if dates else None, "date_max": max(dates) if dates else None,
        "invalid_date_lines": bad_dates,
        "unique_equipment": len({r["equipment_id"] for r in rows}),
        "equipment_types": dict(sorted(Counter(r["equipment_type"] for r in rows).items())),
        "unique_units": len({r["unit_code"] for r in rows}),
        "month_counts": dict(sorted(Counter(day[:7] for day in dates).items())),
        "exact_duplicate_extras": len(rows) - len({tuple(r[c] for c in columns) for r in rows}),
        "equipment_date_duplicate_extras": sum(count - 1 for count in key_counts.values()),
        "missing": missing, "missing_by_maintenance_type": grouped, "numeric": numeric,
        "largest_value_examples": largest_examples,
        "target_counts": dict(sorted(labels.items())) if labels is not None else None,
        "target_positive_rate": labels.get("1", 0) / len(rows) if labels is not None and rows else None,
    }
    return profile, rows


def build_profile() -> dict:
    files, all_rows = {}, []
    for filename in EXPECTED:
        files[filename], rows = profile_file(filename)
        all_rows.extend((filename, row) for row in rows)
    history = defaultdict(list)
    for filename, row in all_rows:
        history[row["equipment_id"]].append((filename, row))
    first_missing, nonfirst_missing, gap_mismatch, cumulative_decrease = Counter(), Counter(), Counter(), Counter()
    for records in history.values():
        records.sort(key=lambda item: item[1]["maintenance_date"])
        previous = None
        for filename, row in records:
            raw_gap = row["days_since_last_maintenance"].strip()
            if not raw_gap:
                (first_missing if previous is None else nonfirst_missing)[filename] += 1
            if previous is not None:
                gap = (date.fromisoformat(row["maintenance_date"]) - date.fromisoformat(previous["maintenance_date"])).days
                if raw_gap and float(raw_gap) != gap:
                    gap_mismatch[filename] += 1
                if float(row["cumulative_operating_hours"]) < float(previous["cumulative_operating_hours"]):
                    cumulative_decrease[filename] += 1
            previous = row
    _, this_month = read_csv(ROOT / "data" / "drone_this_month.csv")
    current_counts = Counter(row["equipment_id"] for row in this_month)
    keys = Counter((row["equipment_id"], row["maintenance_date"]) for _, row in all_rows)
    return {
        "schema_version": 1,
        "purpose": "Analysis preparation only; no model scores or held-out labels are available.",
        "files": files,
        "history_checks": {
            "unique_equipment_all_files": len(history),
            "equipment_date_duplicate_extras_all_files": sum(n - 1 for n in keys.values()),
            "first_observed_gap_missing": dict(first_missing),
            "nonfirst_gap_missing": dict(nonfirst_missing),
            "observed_gap_mismatches": dict(gap_mismatch),
            "cumulative_hours_decreases": dict(cumulative_decrease),
        },
        "current_month": {
            "rows": len(this_month), "latest_unique_candidates": len(latest_record_indices(this_month)),
            "equipment_with_multiple_records": sum(count > 1 for count in current_counts.values()),
            "earlier_records_not_used_for_selection": len(this_month) - len(current_counts),
        },
    }


def markdown(profile: dict) -> str:
    lines = ["# 데이터 사전 점검", "", "계산: `python scripts/profile_data.py`. 원본 CSV를 수정하지 않는 준비 단계 점검이다.",
             "", "| 파일 | 행 | 열 | 드론 수 | 기간 | 완전 중복 추가행 | 드론·날짜 중복 추가행 |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for filename, item in profile["files"].items():
        lines.append(f"| {filename} | {item['rows']:,} | {item['column_count']} | {item['unique_equipment']} | "
                     f"{item['date_min']} ~ {item['date_max']} | {item['exact_duplicate_extras']} | "
                     f"{item['equipment_date_duplicate_extras']} |")
    lines += ["", "## 결측과 정답", ""]
    for filename, item in profile["files"].items():
        missing = ", ".join(f"{c}: {n}건" for c, n in item["missing"].items() if n) or "없음"
        target = (f"재고장 {item['target_counts'].get('1', 0)}건 / {item['rows']}건 "
                  f"({item['target_positive_rate']:.2%})") if item["target_counts"] is not None else "정답 열 없음. 성능 계산 불가."
        lines += [f"- `{filename}`: {missing}. {target}"]
    lines += ["", "예방정비의 고장 코드·유형 공란은 정비 구분과 함께 확인한다. 값이 없는 이유에 대한 가설이며 입력 규칙을 직접 확인한 것은 아니다.",
              "이번 달 파일에는 availability_rate 열 자체가 없다. 셀 결측과 열 부재를 구분한다.",
              "", "## 수치 점검", "", "IQR 1.5배 밖의 값은 검토 후보이며 삭제·클리핑 기준으로 자동 적용하지 않는다. 심각도 등급과 정비 성공 이진값에는 IQR을 적용하지 않는다.",
              "", "| 파일 | 컬럼 | 최솟값 | 중앙값 | 최댓값 | IQR 후보 | 음수 | 파싱 오류 | 범위 오류 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for filename, item in profile["files"].items():
        for column, numbers in item["numeric"].items():
            iqr_flags = numbers.get("iqr_flags")
            iqr_display = iqr_flags if iqr_flags is not None else "적용 안 함"
            lines.append(f"| {filename} | {column} | {numbers.get('min', 'n/a')} | {numbers.get('median', 'n/a')} | "
                         f"{numbers.get('max', 'n/a')} | {iqr_display} | "
                         f"{numbers.get('negative', 'n/a')} | {numbers['invalid']} | {numbers.get('domain_invalid', '—')} |")
    lines += ["", "큰 값의 원본 행 위치와 정비 구분은 JSON의 `files.<파일>.largest_value_examples`에 각 3건씩 저장했다."]
    current = profile["current_month"]
    lines += ["", "## 시계열·이번 달", "",
              f"- 이번 달 {current['rows']}행에서 최신 기록 기준 {current['latest_unique_candidates']}대가 후보이다.",
              f"- 여러 정비 기록이 있는 드론은 {current['equipment_with_multiple_records']}대이며, 최신 선정에서 제외되는 이전 기록은 {current['earlier_records_not_used_for_selection']}행이다. 이전 기록은 이력 피처에 활용할 수 있다.",
              "- 아래 수치는 관측된 기록 간 일관성 점검이다. 미래 정보 사용 허가를 뜻하지 않는다.", "", "```json",
              json.dumps(profile["history_checks"], ensure_ascii=False, indent=2), "```", "",
              "## 아직 계산하지 않은 결과", "", "모델 성능, 최적 임계값, 최종 재점검 20대 및 요인별 기여도는 후속 분석에서 검증한다.",
              "원본 해시와 상세 집계는 `reports/data_profile.json`, 보존 기준은 `reports/data_manifest.json`을 참조한다.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="Print compact counts without writing files")
    args = parser.parse_args()
    profile = build_profile()
    if not args.summary:
        directory = ROOT / "reports"
        directory.mkdir(exist_ok=True)
        (directory / "data_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (directory / "data_profile.md").write_text(markdown(profile), encoding="utf-8")
    for name, item in profile["files"].items():
        print(f"{name}: rows={item['rows']}, equipment={item['unique_equipment']}, "
              f"duplicates={item['exact_duplicate_extras']}, target={'present' if item['target_counts'] else 'absent'}")
    print("current_month:", json.dumps(profile["current_month"], ensure_ascii=False))
    if not args.summary:
        print("Saved reports/data_profile.json and reports/data_profile.md")


if __name__ == "__main__":
    main()
