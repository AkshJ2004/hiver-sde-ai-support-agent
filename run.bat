@echo off
echo ========================================================
echo   Starting AppleSupport AI Copilot
echo   Web UI will be available at http://localhost:8000
echo ========================================================
start http://localhost:8000
uv run python -m uvicorn server:app --port 8000
