# FEAK-Agent repository instructions

작업을 시작하기 전에 `feak_tc_docs/CLAUDE.md`를 읽고 따른다.

## 필수 Git/GitHub 흐름

- 사용자가 명시적으로 요청하지 않는 한 `main`에 직접 push하지 않는다.
- topic branch에서 변경하고 검증한 뒤 해당 브랜치만 원격에 push한다.
- `main` 반영은 GitHub Pull Request로 진행한다. PR 생성과 병합은 검증이 끝난 뒤 에이전트가 `gh` CLI로 직접 수행한다.
