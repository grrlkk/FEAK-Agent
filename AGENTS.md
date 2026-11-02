# FEAK-Agent repository instructions

작업을 시작하기 전에 `README.md`의 구현 위치와 실행 안내를 읽는다.

- 기본 작업 폴더는 `/home/chanwoo/FEAK-Agent` 하나다. 별도 실험 worktree는 사용자가 요청할 때만 만든다.
- 총괄·글수정·한국어 에이전트와 corruption 생성 코드의 공통 의존성을 보존한다.
- LIFT 구현은 `/home/chanwoo/LIFT`와 `grrlkk/LIFT`이며 FEAK 3에이전트와 구분한다.
- 실험 보고서와 과거 지시 문서를 다시 쌓지 않는다. 필요한 사용 안내는 README에 반영한다.
- 원문·데이터·모델·JSON 실행 기록·키·논문 PDF와 백업은 명시적 업로드 요청 없이 커밋하지 않는다.
- 정리 전 복구 자료는 `/home/chanwoo/.local/share/feak-cleanup/20261102-134040/`에 있다.
- 검증은 기존 단위 테스트를 사용한다. 실제 모델 실행이나 유료 실험은 정리 검증에 포함하지 않는다.

## 필수 Git/GitHub 흐름

- 사용자가 명시적으로 요청하지 않는 한 `main`에 직접 push하지 않는다.
- topic branch에서 변경하고 검증한 뒤 해당 브랜치만 원격에 push한다.
- `main` 반영은 GitHub Pull Request로 진행한다. PR 생성과 병합은 검증이 끝난 뒤 에이전트가 `gh` CLI로 직접 수행한다.
