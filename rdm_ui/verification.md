# 스타일 적용·검증 기록

## 현재 재분석과 통합 검증 (2026-11-17)

권장 진입점 app_styled.py와 기존 배포 app.py를 rdm_ui.run_dashboard()에 연결했다. 공통 화면은 rdm_ui/dashboard.py에서 관리한다. 분석 계산·필터·정원·위험 기준·다운로드를 공유하며 화면에서 모델을 재학습하지 않는다.

- N/A·비용 사분위 처리 결과, 12개 한글 모델·피처 구성과 영어 평가 용어를 표시했다. 최종 입력 피처는 23개다.
- 기본 상태 20/81/31/11, 기준 29.85%, 600만원 비교 기준 13.74%를 새 산출물과 대조했다.
- 두 진입점 각각 6개 기능 시나리오를 포함해 필수 검사 7/7·전체 테스트 38개를 통과했다. 전체 수정 Python 타입 검사도 통과했다.
- 실제 Streamlit 실행 서버에서 각 진입점 8개 경우, 총 16개 검사를 통과했다. 모델/비용 표·피처/사분위 표·감사 CSV, 정원 독립성·수동 기준·복원을 확인했고 실행 로그에 새 경고·예외가 없었다.
- 사용자 미리보기 http://127.0.0.1:8517 은 health=ok, Agent API=disabled를 확인했다. 임시 검증 서버 8766/8767은 포트·명령줄을 대조하여 종료했고 기존 8503/8517은 유지했다.
- 근거는 ../reports/app_runtime_check.json, ../reports/check_result.json이다. 브라우저 호스트가 없고 두 Edge 창에서 failed to activate captured window가 발생하여 새 CSS의 스크린샷·모바일·확대 배치는 검증하지 못했다.

아래는 통합 전 별도 가독성 작업 당시의 기록이다. 당시 앱 파일 배치·검사 횟수는 현재 구조와 구분한다.

---

작업일: 2026-11-17 (Asia/Seoul). 요청 범위는 다른 Codex의 분석 작업과 분리한 가독성 보완이다.

## 변경

새 파일 `app_styled.py`, `rdm_ui/__init__.py`, `rdm_ui/readability.css`, 이 폴더의 문서만 추가했다. 원본 앱·헬퍼·테마 설정·패키지 목록·학습 코드·모델·분석 산출물은 편집하지 않았다. 공동 작업 중인 Git 인덱스·브랜치·커밋도 조작하지 않았다. 진행 기록도 공유 분석 로그 대신 이 파일에 분리했다.

청록·남색 색상을 유지하며 제목 크기와 줄바꿈, 설명문 대비·줄간격, 활성 화면과 필터 구분, 숫자 폭, 키보드 포커스를 조정했다. 기본 위젯·표·차트·경고의 동작을 재구현하지 않았다. 스타일은 공개 컨테이너 키로 범위를 제한했고, 사이드바는 그 키가 존재할 때만 HTML의 `aria-label="Sidebar"`를 통해 적용한다.

## 실제 검증

| 검사 | 결과 |
| --- | --- |
| `.venv\Scripts\ty.exe check app_styled.py rdm_ui` | 통과, 진단 없음 |
| 실제 Streamlit 1.67.0 agent API | 10회 모두 `ready`, 예외 없음 |
| 정원 20→5, 수동 위험 기준 100%, 빈 검색, 복원, 3개 화면 전환 | 정상; 정원과 위험 표시 독립성 유지 |
| 현재 목록 CSV·공식 계획 CSV | 20개 고유 드론, 공식 계획 파일 바이트 일치 |
| 기존 `scripts/test_app.py`의 6개 시나리오를 스타일 진입점에 재사용 | 6/6 통과, 요약으로 돌아가는 콜백·정렬 복원 포함 |
| 분리한 복사본의 `scripts/check.py --predictions reports/test_predictions_2025.csv` | 7/7 항목, 테스트 32개 통과 |
| 검증 서버 콘솔 | 새 앱 예외나 사용 중단 경고 없음 |

전체 검사는 `.tmp/ui_preview/check_snapshot`에 당시 파일을 복사한 뒤 원래 프로젝트의 Python 환경으로 실행했다. `check.py`가 생성하는 `reports/check_result.json`도 복사본 안에만 기록했다. AppTest의 bare-mode `missing ScriptRunContext` 메시지는 테스트 환경 경고이며 실행 서버에서는 나타나지 않았다. 임시 검증 스크립트·API 응답·전체 검사 결과는 Git에서 제외되는 `.tmp/ui_preview/`에 있다.

CSS에 지정한 색상의 계산상 대비는 본문 7.38:1, 사이드바 설명 6.77:1, 활성 화면의 흰 글자 4.75:1, 활성 필터 6.99:1이다. 실제 렌더링 측정이나 전체 접근성 감사 결과는 아니다.

## 화면 확인 범위와 한계

기존 웹페이지의 접근성 트리에서 화면 구조·문구·위젯을 확인했다. 브라우저 연결 서비스가 없고 Windows 창 활성화도 `failed to activate captured window`로 실패해, 새 CSS의 스크린샷·모바일 배치·브라우저 확대 시 배치는 육안 확인하지 못했다. 이 항목을 확인 완료로 간주하지 않는다.

검증 API를 켠 임시 서버는 종료하고, API를 끈 확인용 서버를 `http://127.0.0.1:8517`에 별도로 시작했다. 기존 8503 서버와 배포된 웹사이트는 변경하지 않았다. 최종 육안 확인은 이 주소에서 할 수 있고, 이후 실행 방법은 [README.md](README.md)를 따른다.

다른 Codex가 앱이나 분석 결과를 수정하는 동안에도 이 진입점은 원본을 읽는다. 이번 결과는 검증 당시의 파일에 대한 것이며, 이후 원본 위젯 키가 바뀌면 해당 CSS와 화면을 다시 확인한다.

참고 지침: 설치된 `developing-with-streamlit`의 theme/design/layout/selection/testing 문서와 [Vercel web-design-guidelines](https://github.com/vercel-labs/agent-skills/tree/main/skills/web-design-guidelines).
