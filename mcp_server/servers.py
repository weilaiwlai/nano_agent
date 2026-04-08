"""
MCP Servers Management
MCP服务器管理 - 负责启动和管理MCP服务器
"""

import asyncio
import json
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path


class MCPServer:
    """MCP服务器实例"""
    
    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.process: Optional[subprocess.Process] = None
        self._running = False
    
    async def start(self):
        """启动MCP服务器"""
        if self._running:
            return
        
        # 合并环境变量
        full_env = {**subprocess.os.environ, **self.env}
        
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self._running = True
            print(f"[MCP Server:{self.name}] Started with PID {self.process.pid}")
            
        except Exception as e:
            print(f"[MCP Server:{self.name}] Failed to start: {e}")
            raise
    
    async def stop(self):
        """停止MCP服务器"""
        if not self._running or not self.process:
            return
        
        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
        
        self._running = False
        print(f"[MCP Server:{self.name}] Stopped")
    
    async def send_request(self, method: str, params: Dict = None) -> Dict:
        """
        发送JSON-RPC请求到MCP服务器
        
        注意：这是简化实现，实际应该使用更完善的JSON-RPC协议
        """
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        if not self.process or not self._running:
            return {"error": "Server not running"}
        
        try:
            # 发送请求
            request_str = json.dumps(request) + '\n'
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()
            
            # 读取响应（简化实现）
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=30
            )
            
            if response_line:
                response = json.loads(response_line.decode())
                return response.get('result', {})
            else:
                return {"error": "No response"}
                
        except Exception as e:
            return {"error": str(e)}
    
    @property
    def is_running(self) -> bool:
        return self._running


class MCPServerManager:
    """MCP服务器管理器"""
    
    def __init__(self, config: List[Dict]):
        self.config = config
        self.servers: Dict[str, MCPServer] = {}
    
    async def initialize(self):
        """初始化并启动所有配置的MCP服务器"""
        for server_config in self.config:
            name = server_config.get('name')
            command = server_config.get('command')
            args = server_config.get('args', [])
            env = server_config.get('env')
            
            server = MCPServer(name, command, args, env)
            
            try:
                await server.start()
                self.servers[name] = server
            except Exception as e:
                print(f"[MCP Manager] Failed to start server {name}: {e}")
        
        print(f"[MCP Manager] Initialized {len(self.servers)} servers")
    
    async def shutdown(self):
        """关闭所有MCP服务器"""
        for name, server in self.servers.items():
            await server.stop()
        
        self.servers.clear()
        print("[MCP Manager] All servers stopped")
    
    def get_server(self, name: str) -> Optional[MCPServer]:
        """获取指定名称的服务器"""
        return self.servers.get(name)
    
    def list_servers(self) -> List[Dict]:
        """列出所有服务器状态"""
        return [
            {
                "name": name,
                "running": server.is_running,
                "pid": server.process.pid if server.process else None
            }
            for name, server in self.servers.items()
        ]
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict = None) -> Dict:
        """
        调用MCP服务器工具
        
        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 工具参数
        
        Returns:
            工具执行结果
        """
        server = self.get_server(server_name)
        if not server:
            return {"error": f"Server {server_name} not found"}
        
        return await server.send_request(
            method=f"tools/{tool_name}",
            params={"arguments": arguments or {}}
        )
