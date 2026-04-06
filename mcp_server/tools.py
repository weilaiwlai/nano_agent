"""MCP 工具注册模块。"""

from mcp.server.fastmcp import FastMCP

from database import query_database_tool, upsert_user_setting_tool
from email_service import send_report_tool


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
    
    # 将工具函数添加到字典中
    tools_dict["query_database"] = query_database
    tools_dict["send_report"] = send_report
    tools_dict["upsert_user_setting"] = upsert_user_setting
    
    return tools_dict