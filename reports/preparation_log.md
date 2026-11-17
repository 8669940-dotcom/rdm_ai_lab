# RDM 분석 진행 기록

## 2026-11-17 분석 준비

요청 범위: README의 목적에 맞는 분석을 준비하기 위해 AGENTS.md, 프로젝트 스킬과 계산·문제 해결 도구를 정리한다. 최종 모델·앱·보고서는 후속 단계다.

### 입력과 확인

- 업무 근거: `README.txt`의 분석 목적·기간·제출물·비용 가정.
- 입력: `data/drone_train.csv`, `data/drone_test.csv`, `data/drone_this_month.csv`.
- 원본 보존 기준: `reports/data_manifest.json`의 SHA-256. 원본 수정 없이 읽기 전용으로 점검했다.
- 실행 환경: 로컬 Python 3.12.10. 준비 스크립트는 표준 라이브러리만 사용한다.

### 실제 실행한 데이터 점검

명령: `python scripts/profile_data.py`.

| 판단에 사용한 결과 | 근거 파일·필드 | 해석/결정 |
| --- | --- | --- |
| 학습 2,346행, 시험 1,241행, 이번 달 106행 | `data_profile.json: files.<파일>.rows` | README 행 수와 일치 |
| 학습 재고장 525건 | `files.drone_train.csv.target_counts` | 학습 파일에서만 성능 검증 가능 |
| 세 파일 완전 중복·드론/정비일 중복 추가행 0 | `files.<파일>.exact_duplicate_extras`, `equipment_date_duplicate_extras` | 드론 ID 반복을 중복 제거 사유로 쓰지 않음 |
| 이번 달 81대, 복수 정비 드론 18대, 이전 기록 25행 | `current_month` | 선정은 최신 81대에서, 이전 기록은 이력 피처에 사용 가능 |
| 예방정비의 고장 코드·유형 공란 | `files.<파일>.missing_by_maintenance_type` | 구조적 공란 후보로 검토, 발생 원인을 확정한 것은 아님 |
| 학습 경과일 공란 150건 | `files.drone_train.csv.missing`, `history_checks` | 관측 첫 정비인지 연결 점검, 0일로 일괄 대체하지 않음 |
| 이번 달 availability_rate 열 부재 | `files.drone_this_month.csv.columns` | 월말 정보 누출 가능성과 함께 기본 제외 근거로 기록 |
| 학습 운용중단 최댓값 999시간·정비 비용 최댓값 19,000만원 | `files.drone_train.csv.numeric` / `largest_value_examples` | 단위·입력 오류 후보 검토 필요. 아직 삭제·클리핑하지 않음 |

### 준비한 기준과 도구

- `AGENTS.md`: 원본 보존, 시간 분할·과거 이력, 시험 정답 부재, 정원/임계값 분리, 증거 기록과 검증 명령.
- `analysis_config.json`: 2026-01·20대·300/600만원·60만원·초기 검증 시작 제안. 임계값은 미선정(`null`).
- `.agents/skills/rdm-analysis/SKILL.md`: 요약 우선 확인·기존 계산 재사용·필요한 참조만 읽는 작업 절차.
- 스킬 참조 문서: 처리 가설·피처·비교·임계값·제출물·환경/인코딩/키 오류 해결.
- `scripts/profile_data.py`: 원본 분포·결측·중복·시계열 점검 JSON/Markdown 생성.
- `scripts/rdm_common.py`: 최신 행·확률 정렬·비용 계산, 임의 임계값·모델 가중치 없음.
- `scripts/check.py`: 원본 해시·계약·프로필 재현·계산 테스트와 후속 제출물 점검. 결과는 `reports/check_result.json`.

### 검증과 후속 단계

초기 실행에서 준비 검사 6/6과 계산 테스트 7개가 통과했다. 스킬은 `skill-creator/scripts/quick_validate.py .agents/skills/rdm-analysis`로 검증하여 `Skill is valid!`를 확인했다.

이후 수치 요약에서 등급·이진 컬럼의 IQR 표시를 제외하고, 큰 값 3건씩의 원본 위치를 추가했다. 제출물 검사에 대해서도 전체 키 보존·누락·중복 대체·비유한 확률을 검증하는 임시 데이터 테스트를 추가했다.

최종 재실행:

```powershell
& 'C:\Users\86699\AppData\Local\Programs\Python\Python312\python.exe' -X utf8 scripts\profile_data.py
& 'C:\Users\86699\AppData\Local\Programs\Python\Python312\python.exe' -X utf8 scripts\check.py
```

| 검증 | 실제 결과 |
| --- | --- |
| 원본 CSV 3개 SHA-256 | 모두 최초 해시와 일치 |
| 스키마·행 수·기간·정답 유무 | 통과 |
| 저장 집계와 새 계산 대조 | 일치 |
| 최신 기록·동률·비용 경계·정원 독립·확률 검사 | 테스트 통과 |
| 제출물 키 전량 보존·누락·중복 대체·비유한 확률 | 테스트 통과 |
| 전체 테스트 | 11개 통과 |
| 준비 검사 | 6/6 통과, 종료 코드 0 |

구조화된 최종 결과: `reports/check_result.json`. 스킬 검증은 기존 `pdm_ai_lab/.venv`의 Python으로 공식 검증기를 실행했으며 환경이나 패키지를 변경하지 않았다. 이 컴퓨터의 실행 도구에서는 `python`이 PATH에 없어 위의 설치된 실행 파일 경로를 사용했다. 준비 검사 통과는 모델 성능·실습 전체 완료를 의미하지 않는다. 제출물 검사 테스트의 고정 확률은 임시 파일에만 사용했으며 실제 예측 제출 파일은 생성하지 않았다.

후속 순서: 처리 기준 확정, 과거 이력 피처, 학습 내부 시간 분할, 기준선·모델 비교, 검증 비용으로 임계값 선정, 시험 전량 예측, 이번 달 20대 및 요인 설명, 앱/보고서 수치 대조.

미확정 사항: 강사 월 20대 적중의 채점 단위(정비 건/고유 드론), 실제 결측 생성 규칙, 검증으로 선택할 모델·전처리·임계값. 시험 성능은 강사 채점 전 알 수 없다.

## 후속 작업 기록 형식

다음 작업부터 날짜·단계별로 입력/버전, 실행 명령, 결과 파일·필드, 판단과 대안, 검증 결과, 남은 작업을 이어 적는다. 계산하지 않은 결과를 채우거나 이전 실패 기록을 지워 완료처럼 보이게 하지 않는다.

## 2026-11-17 README 검토 후 분석·앱 착수

- 사용자 승인 범위: 데이터 분석·앱 구현·보고서 작성용 진행 기록과 Git 형상관리. 시작 상태 `main...origin/main`, 작업 트리 깨끗함, 커밋 `34d569d`.
- `reports/README_review.md`에 검토하고 `README.txt` 6절에 실행 정의를 보완했다. 원문 업무 목적과 CSV는 보존했다. `AGENTS.md`의 준비 단계 범위를 현재 요청에 맞게 갱신했다.
- 원본 해시·계약·집계 재현과 기존 계산/제출 테스트를 재실행했다: `scripts/check.py` 6/6 통과, 테스트 11개 통과.
- 별도 `.venv`에 `requirements.txt`의 버전을 설치한다. 초기 모델 비교는 학습 파일 내 2024-01~05, 추가 평가는 2024-07~12로 분리한다. 두 학습 경계 모두 정답 30일 성숙 조건을 적용한다. 6월 결과로 7월 예측 모델을 선택하지 않는다.
- 모델 비교·임계값은 앞 검증에서 고정하고 뒤 검증은 성능 확인에 쓴다. 2025년 제출 모델은 2025-01-01에 정답 관측이 끝난 학습 행만, 2026년 운영 모델은 제공된 학습 정답 전체 중 운영 시점에 성숙한 행만 사용한다.
