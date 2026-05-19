@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

echo.
echo ================================================================
echo   HalluGuard -- Demo de bout en bout
echo   Projet 15 -- ENSAM Meknes 4A -- Prof. Hajji Tarik
echo ================================================================
echo.

REM ----------------------------------------------------------------
REM [1/5] Activation du venv
REM ----------------------------------------------------------------
echo [1/5] Activation du venv...
if not exist "venv\Scripts\activate.bat" (
    echo ERREUR : venv non trouve.
    echo Lancez : python -m venv venv et pip install -r requirements.txt
    pause & exit /b 1
)
call venv\Scripts\activate.bat
echo   OK
for /f "tokens=*" %%v in ('venv\Scripts\python.exe --version 2^>^&1') do echo   %%v
echo.

REM ----------------------------------------------------------------
REM [2/5] Serveur MCP en arriere-plan
REM ----------------------------------------------------------------
echo [2/5] Demarrage du serveur MCP HalluGuard en arriere-plan...
start /B "" venv\Scripts\python.exe src\mcp_servers\halluguard_mcp.py
timeout /t 2 /nobreak > nul
echo   OK -- Serveur : mcp://halluguard/stdio ^(protocole MCP/stdio^)
echo   Tools : verify_claim ^| check_temporal_coherence ^| add_fact ^| get_belief_state
echo.

REM ----------------------------------------------------------------
REM [3/5] Pipeline sur 3 questions de demonstration
REM ----------------------------------------------------------------
echo [3/5] Pipeline RAG + HalluGuard sur 3 questions de demonstration...
echo ----------------------------------------------------------------
venv\Scripts\python.exe demo_pipeline.py 2>nul
if errorlevel 1 (
    echo.
    echo ERREUR lors du pipeline -- relance avec details :
    venv\Scripts\python.exe demo_pipeline.py
    pause & exit /b 1
)

REM ----------------------------------------------------------------
REM [4/5] Log HalluGuard (echanges A2A + appels MCP)
REM ----------------------------------------------------------------
echo.
echo [4/5] Log HalluGuard -- derniers echanges A2A
echo ----------------------------------------------------------------
venv\Scripts\python.exe demo_logs.py 2>nul

REM ----------------------------------------------------------------
REM [5/5] Metriques finales du benchmark
REM ----------------------------------------------------------------
echo.
echo [5/5] Metriques finales -- HaluEval-Agentic 60 scenarios
echo ----------------------------------------------------------------
venv\Scripts\python.exe demo_metrics.py 2>nul

echo.
echo ================================================================
echo   Demo terminee avec succes.
echo   Logs : results\logs\a2a_exchanges.log
echo          results\logs\mcp_calls.log
echo          results\logs\halluguard.log
echo ================================================================
echo.
pause
