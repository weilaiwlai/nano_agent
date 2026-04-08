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
    async def search(query: str) -> str:
        """网络搜索关键字查询信息"""
        return await search_tool(query)
           
    # 将工具函数添加到字典中
    tools_dict["search"] = search
    tools_dict["query_database"] = query_database
    tools_dict["send_report"] = send_report
    tools_dict["upsert_user_setting"] = upsert_user_setting
    # tools_dict["geocoding"] = geocoding
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