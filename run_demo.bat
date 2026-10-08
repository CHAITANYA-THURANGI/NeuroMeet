@echo off
echo Running NeuroMeet CLI Meeting Intelligence Demo...
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe scripts\demo_cli.py --scenario tech_postmortem
) else (
    python scripts\demo_cli.py --scenario tech_postmortem
)
pause
