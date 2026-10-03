@echo off
title GPA RAG Backend Server
cd /d "%~dp0"
echo ========================================================
echo   Starting GPA RAG Backend Server (.venv)
echo ========================================================
"..\.venv\Scripts\python.exe" server.py
pause
