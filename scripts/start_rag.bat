@echo off
chcp 65001 >nul
title RAG Travel Guide Service (KEEP THIS WINDOW OPEN)
cd /d "%~dp0..\rag-system"
echo ============================================================
echo  RAG Travel Guide Service  ^|  port 8010
echo  KEEP THIS WINDOW OPEN while using the travel assistant.
echo  Close this window (or Ctrl+C) to stop the service.
echo ============================================================
echo.
"..\.venv\Scripts\python.exe" rag_api.py
echo.
echo [Service stopped] Press any key to close...
pause >nul
