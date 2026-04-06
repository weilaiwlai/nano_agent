@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   NanoAgent Startup Script
echo ========================================
echo.

echo [1/3] Starting MCP Service (port 8000)...
start "NanoAgent MCP" cmd /k "cd /d "%~dp0mcp_server" && uvicorn main:app --host 0.0.0.0 --port 8000"
timeout /t 5 /nobreak >nul

curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ERROR: MCP Service failed to start
    goto :error
) else (
    echo OK: MCP Service started successfully
)

echo.
echo [2/3] Starting Agent Service (port 8080)...
start "NanoAgent Agent" cmd /k "cd /d "%~dp0agent_service" && uvicorn main:app --host 0.0.0.0 --port 8080"
timeout /t 5 /nobreak >nul

curl -s http://localhost:8080/health >nul 2>&1
if errorlevel 1 (
    echo ERROR: Agent Service failed to start
    goto :error
) else (
    echo OK: Agent Service started successfully
)

echo.
echo [3/3] Starting Frontend Service (port 8501)...
start "NanoAgent Frontend" cmd /k "cd /d "%~dp0frontend" && streamlit run app.py"
timeout /t 8 /nobreak >nul

echo.
echo ========================================
echo   SUCCESS: All services started!
echo ========================================
echo.
echo Service URLs:
echo   Frontend: http://localhost:8501
echo   Agent API: http://localhost:8080
echo   MCP Service: http://localhost:8000
echo.
echo Press any key to open the frontend in browser...
pause >nul

start http://localhost:8501

echo.
echo Services are running in background.
echo Close the command windows to stop services.
echo.
pause
exit /b 0

:error
echo.
echo ERROR: Service startup failed!
echo Please check:
echo   1. Database connections
echo   2. Port availability  
echo   3. Dependencies installation
echo   4. Environment configuration
echo.
pause
exit /b 1