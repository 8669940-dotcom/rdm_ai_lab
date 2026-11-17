# 데이터 사전 점검

계산: `python scripts/profile_data.py`. 원본 CSV를 수정하지 않는 준비 단계 점검이다.

| 파일 | 행 | 열 | 드론 수 | 기간 | 완전 중복 추가행 | 드론·날짜 중복 추가행 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| drone_train.csv | 2,346 | 18 | 150 | 2023-01-01 ~ 2024-12-31 | 0 | 0 |
| drone_test.csv | 1,241 | 17 | 150 | 2025-01-01 ~ 2025-12-31 | 0 | 0 |
| drone_this_month.csv | 106 | 16 | 81 | 2026-01-01 ~ 2026-01-31 | 0 | 0 |

## 결측과 정답

- `drone_train.csv`: fault_code: 1707건, fault_category: 1707건, days_since_last_maintenance: 150건. 재고장 525건 / 2346건 (22.38%)
- `drone_test.csv`: fault_code: 876건, fault_category: 876건. 정답 열 없음. 성능 계산 불가.
- `drone_this_month.csv`: fault_code: 62건, fault_category: 62건. 정답 열 없음. 성능 계산 불가.

예방정비의 고장 코드·유형 공란은 정비 구분과 함께 확인한다. 값이 없는 이유에 대한 가설이며 입력 규칙을 직접 확인한 것은 아니다.
이번 달 파일에는 availability_rate 열 자체가 없다. 셀 결측과 열 부재를 구분한다.

## 수치 점검

IQR 1.5배 밖의 값은 검토 후보이며 삭제·클리핑 기준으로 자동 적용하지 않는다. 심각도 등급과 정비 성공 이진값에는 IQR을 적용하지 않는다.

| 파일 | 컬럼 | 최솟값 | 중앙값 | 최댓값 | IQR 후보 | 음수 | 파싱 오류 | 범위 오류 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| drone_train.csv | equipment_age_years | 0.6 | 4.4 | 7.9 | 0 | 0 | 0 | — |
| drone_train.csv | severity | 0.0 | 0.0 | 5.0 | 적용 안 함 | 0 | 0 | 0 |
| drone_train.csv | downtime_hours | 0.5 | 3.5 | 999.0 | 425 | 0 | 0 | — |
| drone_train.csv | repair_duration_hours | 0.2 | 2.0 | 55.9 | 345 | 0 | 0 | — |
| drone_train.csv | parts_replaced_count | 0.0 | 0.0 | 10.0 | 253 | 0 | 0 | 0 |
| drone_train.csv | maintenance_cost | 8.0 | 23.0 | 19000.0 | 434 | 0 | 0 | — |
| drone_train.csv | maintenance_success | 0.0 | 1.0 | 1.0 | 적용 안 함 | 0 | 0 | 0 |
| drone_train.csv | cumulative_operating_hours | 140.3 | 1363.4 | 3542.6 | 8 | 0 | 0 | — |
| drone_train.csv | days_since_last_maintenance | 1.0 | 51.0 | 94.0 | 0 | 0 | 0 | — |
| drone_train.csv | availability_rate | 0.3 | 0.965 | 0.998 | 307 | 0 | 0 | 0 |
| drone_test.csv | equipment_age_years | 2.5 | 5.8 | 8.9 | 0 | 0 | 0 | — |
| drone_test.csv | severity | 0.0 | 0.0 | 5.0 | 적용 안 함 | 0 | 0 | 0 |
| drone_test.csv | downtime_hours | 0.5 | 3.6 | 61.8 | 196 | 0 | 0 | — |
| drone_test.csv | repair_duration_hours | 0.2 | 2.1 | 41.9 | 186 | 0 | 0 | — |
| drone_test.csv | parts_replaced_count | 0.0 | 0.0 | 9.0 | 146 | 0 | 0 | 0 |
| drone_test.csv | maintenance_cost | 8.0 | 24.0 | 611.0 | 158 | 0 | 0 | — |
| drone_test.csv | maintenance_success | 0.0 | 1.0 | 1.0 | 적용 안 함 | 0 | 0 | 0 |
| drone_test.csv | cumulative_operating_hours | 758.1 | 2131.6 | 4270.7 | 10 | 0 | 0 | — |
| drone_test.csv | days_since_last_maintenance | 1.0 | 51.0 | 94.0 | 0 | 0 | 0 | — |
| drone_test.csv | availability_rate | 0.3 | 0.965 | 0.997 | 158 | 0 | 0 | 0 |
| drone_this_month.csv | equipment_age_years | 3.5 | 6.4 | 9.0 | 0 | 0 | 0 | — |
| drone_this_month.csv | severity | 0.0 | 0.0 | 5.0 | 적용 안 함 | 0 | 0 | 0 |
| drone_this_month.csv | downtime_hours | 0.6 | 4.0 | 56.0 | 16 | 0 | 0 | — |
| drone_this_month.csv | repair_duration_hours | 0.3 | 2.4 | 26.6 | 18 | 0 | 0 | — |
| drone_this_month.csv | parts_replaced_count | 0.0 | 0.0 | 7.0 | 4 | 0 | 0 | 0 |
| drone_this_month.csv | maintenance_cost | 8.0 | 25.5 | 452.0 | 5 | 0 | 0 | — |
| drone_this_month.csv | maintenance_success | 0.0 | 1.0 | 1.0 | 적용 안 함 | 0 | 0 | 0 |
| drone_this_month.csv | cumulative_operating_hours | 1000.8 | 2410.75 | 4172.1 | 2 | 0 | 0 | — |
| drone_this_month.csv | days_since_last_maintenance | 1.0 | 44.0 | 85.0 | 0 | 0 | 0 | — |

큰 값의 원본 행 위치와 정비 구분은 JSON의 `files.<파일>.largest_value_examples`에 각 3건씩 저장했다.

## 시계열·이번 달

- 이번 달 106행에서 최신 기록 기준 81대가 후보이다.
- 여러 정비 기록이 있는 드론은 18대이며, 최신 선정에서 제외되는 이전 기록은 25행이다. 이전 기록은 이력 피처에 활용할 수 있다.
- 아래 수치는 관측된 기록 간 일관성 점검이다. 미래 정보 사용 허가를 뜻하지 않는다.

```json
{
  "unique_equipment_all_files": 150,
  "equipment_date_duplicate_extras_all_files": 0,
  "first_observed_gap_missing": {
    "drone_train.csv": 150
  },
  "nonfirst_gap_missing": {},
  "observed_gap_mismatches": {},
  "cumulative_hours_decreases": {}
}
```

## 아직 계산하지 않은 결과

모델 성능, 최적 임계값, 최종 재점검 20대 및 요인별 기여도는 후속 분석에서 검증한다.
원본 해시와 상세 집계는 `reports/data_profile.json`, 보존 기준은 `reports/data_manifest.json`을 참조한다.
