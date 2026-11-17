# rdm_ai_lab — 드론 예측정비

교육용 가상 정비 데이터로 정비 직후 30일 이내 재고장을 예측하고, **2026년 1월 고유 드론 81대 중 재점검 20대**를 선정한다. 원 과제와 검토 보완 기준은 [README.txt](README.txt), 작업 규칙은 [AGENTS.md](AGENTS.md)를 따른다.

## 설치와 실행

Python 3.12 기준이다. 프로젝트 폴더의 PowerShell에서 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

분석 산출물이 Git에 포함되어 있어 앱 실행 시 재학습하지 않는다. API 키나 외부 데이터베이스는 필요 없다. `python`을 찾지 못하면 설치된 Python 실행 파일 또는 `py -3.12`를 사용한다. 가상환경 활성화 없이 위 실행 파일 경로로 실행할 수 있다.

## 분석 재현과 검증

```powershell
.\.venv\Scripts\python.exe scripts/profile_data.py
.\.venv\Scripts\python.exe scripts/train.py
.\.venv\Scripts\python.exe scripts/build_report.py
.\.venv\Scripts\python.exe scripts/build_briefing.py
.\.venv\Scripts\python.exe scripts/check.py --predictions reports/test_predictions_2025.csv
```

개발 검증에는 다음을 추가한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\ty.exe check app.py app_helpers.py rdm_model.py scripts/train.py scripts/build_report.py scripts/build_briefing.py scripts/test_briefing.py
```

`check.py`는 원본 해시·데이터 계약·집계 재현과, 분석 결과가 있을 때 시간 경계·모델 예측 재현·앱 동작 테스트를 수행한다. 모델 성능은 저장된 내부 시간 검증 결과이며 실제 부대 안전성이나 예방 효과를 보증하지 않는다.

## 화면 진행 순서

1. **재점검 계획**: 전체 81대에서 선정한 순위·위험 표시·대기 대수를 확인한다. 유형/부대/정비 구분 요약의 구분을 누르면 전체 후보 상세로 이동하며 돌아가기 버튼으로 요약을 복원한다.
2. 드론·유형·부대·최근 정비일·정비 구분·확률로 정렬한다. 필터는 전체 순위와 명단을 바꾸지 않는다. 드론 상세에서 관측 이력·위험 민감도·권고를 본다.
3. 정원과 위험 임계값을 바꾸어 시나리오를 살핀다. **공식 20대 계획 다운로드는 기본 산출물**, 현재 목록 다운로드는 화면 조건을 따른다.
4. **모델 검증**: 앞 기간 모델 비교와 별도 하반기 평가, 월20대 기준선, 300/600만원 비용 가정과 임계값을 확인한다.
5. **데이터·과정**: 파일 계약·유형/월/고장별 분포·처리 근거·시간 경계와 보고서 작성용 기록을 확인한다.

## 결과와 근거

- [README 검토표](reports/README_review.md)
- [분석 과정·보고서 작성 근거](reports/analysis_process.md)
- [2026년 1월 재점검 계획](reports/reinspection_plan_2026-01.md)
- [작업 진행·문제 해결 기록](reports/preparation_log.md)
- [기계 판독 분석 결과](reports/analysis_results.json), [원본 점검](reports/data_profile.md), [검증 결과](reports/check_result.json)
- 시험 채점 제출: `reports/test_predictions_2025.csv` (원본 1,241행·3열·정답 없음)

## 지휘관 보고·발표 자료

- [분석결과보고](reports/2026-01_분석결과보고.md): 이전 pdm 보고서 형식의 결론·품질·처리·모델·기준선·비용·한계 보고.
- [5분 지휘관 발표 대본](reports/2026-01_지휘관발표스크립트.md): 화면 조작 순서·발표 대사·12개 예상 질의와 출처·시연 실패 대응.
- [계산식·출처 대조표](reports/계산식_출처_대조표.md): F01–F17 산식, 실제 코드 함수·행 위치, 결과 JSON 필드와 공식 출처.
- `reports/briefing_evidence.json`: AP·PR 면적·ROC·Brier의 독립 수식, 기준선·비용·개별 설명 및 순열 중요도 재검산 결과.
- `python scripts/build_briefing.py`는 검산용 모델을 메모리에서만 재구성한다. 원본 CSV·저장 모델·기존 분석 산출물은 바꾸지 않는다. 보고서용 해시는 `reports/briefing_manifest.json`으로 확인한다.

## 분석 해석 기준

- 2024-01~05에서 모델·임계값을 선택하고 2024-07~12에서 고정 절차를 평가했다. 학습 경계에서는 30일 정답 관측 기간을 제외했다.
- 2025년 예측 모델은 연초에 정답이 성숙한 학습 행으로 고정하고, 2026년 운영 모델은 제공된 학습 정답 전체를 사용한다. 시험 성능은 강사 채점 전에는 제시하지 않는다.
- AP와 사다리꼴 PR-AUC를 구분한다. 월20대 운영은 고유 드론 최신 기록 기준이며 정비 기록별 상위20건 채점 지표도 별도 저장한다.
- 위험 임계값과 정원은 독립적이다. `analysis_config.json`의 임계값 `null`은 검증 비용으로 자동 선정한다는 뜻이며 결과는 `analysis_results.json`의 `threshold_policies`가 소유한다.
- 드론별 설명은 학습 참조값 치환 확률 민감도다. SHAP·합산 가중치·인과관계가 아니다. 최종 정비·운용 판단은 담당자가 한다.

## 형상관리

원본 CSV는 바꾸지 않고 `.gitattributes`로 바이트를 보존한다. `.venv`, 비밀 설정과 임시 실행 로그는 제외한다. 모델·데이터·검증된 분석 산출물은 함께 관리한다. 재학습 후 보고서를 다시 생성하고 필수 검사를 통과한 변경만 커밋한다.
