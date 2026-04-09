"""MCP 工具注册模块。"""

from mcp.server.fastmcp import FastMCP

from database import query_database_tool, upsert_user_setting_tool
from email_service import send_report_tool
# from mcp_server_amap import geocoding,reverse_geocoding,poi_search,weather_query,route_planning
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp_server_time import get_current_time
import os
import json
import asyncio
import logging
from seacrch import search_tool
from filesystem_service import FilesystemService



def register_tools(mcp: FastMCP) -> dict[str, callable]:
    """注册所有MCP工具，并返回工具字典。"""
    tools_dict = {}
    
    @mcp.tool()
    async def query_database(sql: str) -> str:
        """异步执行 SQL，并将结果集以 JSON 返回。"""
        return await query_database_tool(sql)
    
    @mcp.tool()
    async def send_report(email: str, content: str) -> str:
        """发送报告邮件（支持 mock/smtp 双模式），并返回执行结果。"""
        return await send_report_tool(email, content)
    
    @mcp.tool()
    async def upsert_user_setting(user_id: str, setting_key: str, setting_value: str) -> str:
        """受控写工具：仅允许更新白名单用户设置键。"""
        return await upsert_user_setting_tool(user_id, setting_key, setting_value)
    
    @mcp.tool()
    async def search(query: str) -> str:
        """网络搜索关键字查询信息"""
        return await search_tool(query)
    filesystem = FilesystemService(['D:/nano_agent/agentdata'])
    @mcp.tool()
    async def is_path_allowed(path: str) -> str:
        """检查路径是否被允许"""
        return await filesystem.is_path_allowed(path)
    @mcp.tool()
    async def read_file(path: str) -> str:
        """读取文件内容"""
        return await filesystem.read_file(path)
    @mcp.tool()
    async def write_file(path: str, content: str) -> str:
        """写入文件内容"""
        return await filesystem.write_file(path, content)
    @mcp.tool()
    async def create_directory(path: str) -> str:
        """创建目录"""
        return await filesystem.create_directory(path)
    @mcp.tool()
    async def move_file(path: str, new_path: str) -> str:
        """移动文件"""
        return await filesystem.move_file(path, new_path)
    @mcp.tool()
    async def edit_file(path: str, content: str) -> str:
        """编辑文件内容"""
        return await filesystem.edit_file(path, content)
    @mcp.tool()
    async def list_allowed_directories() -> str:
        """返回允许访问的目录列表"""
        return await filesystem.list_allowed_directories()
    # 将工具函数添加到字典中
    tools_dict["search"] = search
    tools_dict["query_database"] = query_database
    tools_dict["send_report"] = send_report
    tools_dict["upsert_user_setting"] = upsert_user_setting
    tools_dict["is_path_allowed"] = is_path_allowed
    tools_dict["read_file"] = read_file
    tools_dict["write_file"] = write_file
    tools_dict["create_directory"] = create_directory
    tools_dict["move_file"] = move_file
    tools_dict["edit_file"] = edit_file
    tools_dict["list_allowed_directories"] = list_allowed_directories
    tools_dict["get_current_time"] = get_current_time
    # tools_dict["reverse_geocoding"] = reverse_geocoding
    # tools_dict["poi_search"] = poi_search
    # tools_dict["weather_query"] = weather_query
    # tools_dict["route_planning"] = route_planning
    # tools_dict["get_current_time"] = get_current_time
    # CONFIG_FILE_PATH="mcp_config.json"
    # if os.path.exists(CONFIG_FILE_PATH):
    #     with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
    #         mcp_config = json.load(f)
    # client = MultiServerMCPClient(mcp_config)
    # 方法1: 直接获取工具
    # tools = await client.get_tools()
    return tools_dict
    # 使用新的API方式创建客户端

    # tools_dict.update(tools)
    # logging.info(f"已注册工具: {tools_dict.keys()}")
    # yield tools_dict