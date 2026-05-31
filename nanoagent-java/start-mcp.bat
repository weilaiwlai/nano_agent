@echo off
chcp 65001 >nul
title NanoAgent MCP Server (Port 8000)

cd /d "%~dp0"

if exist ".env" (
    echo [dotenv] 已加载 .env 配置
) else (
    echo [警告] .env 文件不存在，使用默认配置
    echo   请复制 .env.example 为 .env 并填入配置
    echo.
)

echo ============================================
echo   NanoAgent MCP Server
echo   Port: 8000
echo ============================================

cd nano-mcp-server
call mvn spring-boot:run
pause
