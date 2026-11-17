# RDM 보고용 화면과 가독성 모듈

권장 진입점은 **app_styled.py**다. 기존 배포 진입점 **app.py**도 같은 화면을 사용한다.
두 파일은 rdm_ui.run_dashboard()를 호출하며, 공통 화면은 rdm_ui/dashboard.py, 스타일은 readability.css가 소유한다.
원본·모델·분석 계산은 기존 모듈과 저장 산출물을 사용한다.

프로젝트 루트에서 실행한다.

    .\.venv\Scripts\python.exe -m streamlit run app_styled.py --server.address=127.0.0.1 --server.port=8517 --server.enableAgentApi=false

보고용 확인 주소는 **http://127.0.0.1:8517**이다. 기존 app.py 배포의 진입 파일을 바꾸지 않아도 같은 스타일이 적용된다.

## 현재 재분석 결과 연결

- 시간 999·공란 N/A, 비용 100배 오류 가정·사분위 제한의 처리 기준과 치환 감사 CSV
- 12개 한글 모델·피처 구성, 최종 한글 피처 23개, Precision·Recall·TPR·FPR
- AI 활용 / 데이터 분석 보고서, 5분 발표 대본, 계산식·출처 다운로드
- 기본 선정 20 / 후보 81 / 위험 31 / 정원 밖 위험 11, 기준 29.85%

분석 수치는 analysis_results.json, 전처리 전후 값은 cleaning_audit.csv에서 읽는다.
화면을 다시 실행할 때 모델을 학습하거나 위험 기준을 다시 선정하지 않는다.

## 스타일과 구조

- 공개 key=rdm_readability 컨테이너로 스타일 범위를 정한다.
- 제목·설명 대비·행간, 활성 화면·필터, 숫자 폭과 키보드 포커스를 보완한다.
- 현재 목록·계획·치환 감사·월별 검증·시험 예측 다운로드의 버튼 키를 연결했다.
- 모델 비교표의 긴 한글 구성명은 Streamlit TextColumn의 넓은 열로 표시한다.
- 표와 차트는 Streamlit의 기본 테마를 사용하며 추가 패키지·외부 폰트·JavaScript는 필요 없다.

공통 화면이나 CSS를 수정한 뒤 자동 갱신이 보이지 않으면 Streamlit의 Rerun 또는 브라우저 새로고침으로 반영한다.

## 검증

두 진입점 각각 6개 AppTest 시나리오를 유지한다. 전체 필수 검사 7/7, 테스트 38개, 타입 검사를 통과했다.
실제 Streamlit Agent API에서도 각 진입점 8개 경우, 총 16개 검사를 통과했다.
사용자 확인용 8517 서버는 health=ok, Agent API=disabled다.
브라우저 연결 서비스와 Windows 창 활성화 오류로 새 CSS의 육안·모바일 배치 점검은 완료하지 못했다.
상세 범위와 초기 검증 이력은 [verification.md](verification.md), 현재 실행 결과는 [app_runtime_check.json](../reports/app_runtime_check.json)에 있다.
