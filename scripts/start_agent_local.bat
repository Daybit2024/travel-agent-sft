@echo off
chcp 65001 >nul
title Travel Assistant - LOCAL model qwen3-travel
cd /d "%~dp0.."

set TRAVEL_LLM_BASE_URL=http://localhost:11434/v1
set TRAVEL_LLM_MODEL=qwen3-travel
set TRAVEL_LLM_API_KEY=ollama

curl -s -o nul --max-time 3 http://localhost:11434/api/version
if errorlevel 1 goto NOOLLAMA

curl -s -o nul --max-time 3 http://127.0.0.1:8010/health
if errorlevel 1 goto NORAG
goto RUN

:NOOLLAMA
echo [ERROR] Ollama is not running.
echo Start Ollama first: Start menu -^> Ollama, or run: ollama serve
echo.
pause
exit /b 1

:NORAG
echo [WARN] RAG service on port 8010 is not running - guide search will fail.
echo Start it first with: start_rag.bat
echo.

:RUN
echo ============================================================
echo  Travel Assistant  ^|  LOCAL model: qwen3-travel - CPU, slow
echo  Each round may take 1-3 minutes. Please be patient.
echo ============================================================
echo.
".venv\Scripts\python.exe" -m agent_graph.web_demo_graph
echo.
echo [Assistant exited] Press any key to close...
pause >nul
