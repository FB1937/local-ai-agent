
@echo off
chcp 65001 > nul
setlocal
title Local AI Agent

REM ==========================================
REM LOCAL AI AGENT LAUNCHER
REM ==========================================

cd /d "%~dp0"

echo.
echo ==========================================
echo          LOCAL AI AGENT START
echo ==========================================
echo.

REM ------------------------------------------
REM 1. Python environment
REM ------------------------------------------

echo [1/5] Checking Python environment...

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found.
    goto ERROR
)

REM ------------------------------------------
REM 2. Docker and SearXNG
REM ------------------------------------------

echo.
echo [2/5] Checking Docker...

docker info > nul 2>&1
if not errorlevel 1 goto DOCKER_READY

echo Docker is not running. Starting Docker Desktop...

set "DOCKER_EXE=%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"

if not exist "%DOCKER_EXE%" (
    echo [ERROR] Docker Desktop executable not found.
    echo Expected: %DOCKER_EXE%
    goto ERROR
)

start "" "%DOCKER_EXE%"

echo Waiting for Docker...

for /L %%i in (1,1,60) do (
    docker info > nul 2>&1
    if not errorlevel 1 goto DOCKER_READY
    timeout /t 2 /nobreak > nul
)

echo [ERROR] Docker startup timeout.
goto ERROR

:DOCKER_READY

echo Docker is ready.
echo Starting SearXNG...

set "SEARXNG_DIR=%USERPROFILE%\AI\searxng"

if not exist "%SEARXNG_DIR%" (
    echo [ERROR] SearXNG directory not found.
    goto ERROR
)

cd /d "%SEARXNG_DIR%"

docker compose up -d
if errorlevel 1 (
    echo [ERROR] Failed to start SearXNG.
    goto ERROR
)

echo Waiting for SearXNG...

set "SEARXNG_READY="

for /L %%i in (1,1,30) do (
    curl.exe -fsS "http://localhost:8080/" -o nul
    if not errorlevel 1 (
        set "SEARXNG_READY=1"
        goto SEARXNG_READY
    )
    timeout /t 2 /nobreak > nul
)

:SEARXNG_READY

if not defined SEARXNG_READY (
    echo [ERROR] SearXNG startup timeout.
    goto ERROR
)

echo SearXNG is ready.

REM ------------------------------------------
REM 3. Ollama
REM ------------------------------------------

echo.
echo [3/5] Checking Ollama...

curl.exe -fsS "http://localhost:11434/api/tags" -o nul

if not errorlevel 1 goto OLLAMA_READY

echo Ollama is not running. Starting Ollama...

where ollama > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama command not found.
    goto ERROR
)

start "" /min ollama serve

echo Waiting for Ollama...

set "OLLAMA_READY="

for /L %%i in (1,1,30) do (
    curl.exe -fsS "http://localhost:11434/api/tags" -o nul
    if not errorlevel 1 (
        set "OLLAMA_READY=1"
        goto OLLAMA_STARTED
    )
    timeout /t 2 /nobreak > nul
)

:OLLAMA_STARTED

if not defined OLLAMA_READY (
    echo [ERROR] Ollama startup timeout.
    goto ERROR
)

:OLLAMA_READY

echo Ollama is ready.

REM ------------------------------------------
REM 4. Qwen model
REM ------------------------------------------

echo.
echo [4/5] Checking Qwen 3.5 9B...

ollama list | findstr /i /c:"qwen3.5:9b" > nul

if errorlevel 1 (
    echo [ERROR] Qwen 3.5 9B model not found.
    echo Run: ollama pull qwen3.5:9b
    goto ERROR
)

echo Qwen 3.5 9B is ready.

REM ------------------------------------------
REM 5. Python AI Agent
REM ------------------------------------------

echo.
echo [5/5] Starting Local AI Agent...
echo.

cd /d "%~dp0"

".venv\Scripts\python.exe" main.py

echo.
echo ==========================================
echo           AGENT STOPPED
echo ==========================================
echo.

pause
exit /b 0

REM ------------------------------------------
REM ERROR HANDLER
REM ------------------------------------------

:ERROR

echo.
echo ==========================================
echo              STARTUP FAILED
echo ==========================================
echo.

pause
exit /b 1
