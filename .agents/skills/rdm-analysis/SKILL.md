---
name: rdm-analysis
description: Prepare and analyze the rdm_ai_lab drone maintenance exercise using its README, chronological data, reproducible calculations and evidence logs. Use for data checks, feature preparation, validation, threshold costs and January 2026 selection in this project.
---

# RDM 분석과 재현

프로젝트 루트는 이 파일에서 `../../..`이다. `README.txt`와 `AGENTS.md`를 기준으로 작업한다. 현재 단계와 요청 범위를 먼저 확인하고 모델 성능·앱 완료를 미리 선언하지 않는다.

## 최소 컨텍스트로 시작

1. `reports/data_profile.md`와 `analysis_config.json`을 읽는다. 프로필이 없거나 원본이 바뀌었는지 의심되면 루트에서 `python scripts/profile_data.py --summary`로 확인한다.
2. 전체 CSV를 대화에 붙이지 않는다. 특정 컬럼·기간·오류만 스크립트로 집계하고 상세 결과는 JSON에 둔다. 요약에 없는 숫자는 계산한 후 인용한다.
3. 이미 있는 함수를 재사용한다. `scripts/rdm_common.py`의 `rank_latest(rows, probabilities, capacity)`는 원본 행 순서에 맞는 확률을 받아 최신 드론별로 순위를 낸다. `threshold_cost`는 정원 제한 없는 검증 비용을 계산한다.
4. 현재 작업에 해당하는 [분석 절차와 문제 해결](references/analysis-protocol.md) 부분만 읽는다. 후속 모델·앱의 라이브러리 API는 해당 설치 버전의 공식 문서/전용 스킬을 확인한다.

## 실행과 검증

```powershell
python scripts/profile_data.py
python scripts/check.py
```

준비 도구는 Python 표준 라이브러리만 사용한다. 원본을 읽고 `reports/`의 점검 파일을 갱신한다. `check.py --predictions <파일>`은 후속 시험 제출물의 전체 행·키·날짜·확률을 검사한다.

특정 계산을 임시 대화 코드로 반복하기보다 `scripts/`에 함수로 남긴다. 보고서용 계산은 입력·단위·기간·집계 단위를 명시하고, 최신 행 선정·확률 동률·비용 경계 등 중요한 성질을 작은 예제로 검증한다.

결과는 `reports/preparation_log.md`에 명령, 판단과 근거 파일을 연결해 기록한다. 실행하지 않은 모델 비교, 실제로 없는 시험 정답, 미선정 임계값을 숫자로 채우지 않는다. `python scripts/check.py`가 통과해도 모델 검증 완료로 해석하지 않는다.
