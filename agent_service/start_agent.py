#!/usr/bin/env python3
"""智能体服务启动脚本"""

import os
import uvicorn
from dotenv import load_dotenv
import asyncio

# 加载环境变量
load_dotenv('../.env')

if __name__ == "__main__":
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "8080"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🚀 启动智能体服务: http://{host}:{port}")
    print(f"📊 调试模式: {debug}")
    print("💡 健康检查: http://localhost:8080/health")
    print("🔧 MCP服务连接: http://localhost:8000")
    print("-" * 50)
    
    uvicorn.run("main:app", host=host, port=port, reload=debug,loop=asyncio.SelectorEventLoop)