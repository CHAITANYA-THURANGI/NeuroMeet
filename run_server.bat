@echo off
echo Starting NeuroMeet FastAPI backend & Web Intelligence Studio...
echo URL: http://127.0.0.1:8000
echo Docs: http://127.0.0.1:8000/docs
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
)
pause
