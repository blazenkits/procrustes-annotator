# 사용법

1. [uv](https://github.com/astral-sh/uv)를 다운받습니다. 
```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. 저장된 프로젝트 열기
```bash
uv run procrustes-annotator main.project
```

- (프로젝트가 없는 경우)
```bash
uv run procrustes-annotator