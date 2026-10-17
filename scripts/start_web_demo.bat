@echo off
chcp 65001 >nul
title Travel Assistant - Web Demo (Gradio)
cd /d "%~dp0.."

echo ============================================================
echo  Travel Assistant Web Demo
echo  浏览器打开: http://127.0.0.1:7860   （关闭本窗口即停止）
echo  默认云端模型 qwen-plus（需要 DASHSCOPE_API_KEY）
echo ============================================================
echo.
".venv\Scripts\python.exe" -m agent_graph.web_demo_graph
echo.
echo [Demo exited] Press any key to close...
pause >nul
