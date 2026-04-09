import requests
import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

async def search_tool(query: str) -> str:
    """
    搜索关键字
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "错误：未配置 TAVILY_API_KEY 环境变量，无法进行网络搜索。"
    
    search_tool = TavilySearch(api_key=tavily_api_key)
    try:
        results = search_tool.invoke(query)
        return str(results['results'])
    except Exception as e:
        return f"搜索失败：{str(e)}"
    
if __name__ == "__main__":
    result=search_tool("测试")
    print(result)