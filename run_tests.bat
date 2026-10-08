@echo off
echo Running NeuroMeet automated test suite...
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe -m pytest -v
) else (
    python -m pytest -v
)
pause
