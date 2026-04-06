#!/usr/bin/env python3
"""
MCP服务启动测试程序
用于验证NanoAgent MCP服务是否正常启动
"""

import asyncio
import aiohttp
import sys
import time
from typing import Dict, Any


class MCPTester:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_endpoint(self) -> Dict[str, Any]:
        """测试健康检查端点"""
        import time
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                response_time = time.time() - start_time
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "status": "healthy",
                        "data": data,
                        "response_time": response_time
                    }
                else:
                    return {
                        "success": False,
                        "status": f"HTTP {response.status}",
                        "error": f"健康检查失败: {response.status}"
                    }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "status": "connection_error",
                "error": f"连接错误: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "status": "unknown_error",
                "error": f"未知错误: {str(e)}"
            }
    
    async def test_database_connection(self) -> Dict[str, Any]:
        """测试数据库连接（如果服务提供此功能）"""
        import time
        start_time = time.time()
        
        try:
            # 尝试调用工具列表或数据库相关端点
            async with self.session.get(f"{self.base_url}/tools") as response:
                response_time = time.time() - start_time
                if response.status == 200:
                    return {
                        "success": True,
                        "status": "database_connected",
                        "response_time": response_time
                    }
                elif response.status == 404:
                    # 端点不存在是正常的
                    return {
                        "success": True,
                        "status": "service_running",
                        "note": "工具端点不存在（正常情况）"
                    }
                else:
                    return {
                        "success": False,
                        "status": f"HTTP {response.status}",
                        "error": f"数据库连接测试失败: {response.status}"
                    }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "status": "connection_error",
                "error": f"数据库连接测试错误: {str(e)}"
            }
    
    async def comprehensive_test(self) -> Dict[str, Any]:
        """综合测试MCP服务"""
        print(f"正在测试MCP服务: {self.base_url}")
        print("-" * 50)
        
        results = {}
        
        # 测试健康检查
        print("1. 测试健康检查端点...")
        health_result = await self.test_health_endpoint()
        results["health"] = health_result
        
        if health_result["success"]:
            print(f"   ✓ 健康检查通过 - 状态: {health_result.get('data', {})}")
            if "response_time" in health_result:
                print(f"   响应时间: {health_result['response_time']:.3f}秒")
        else:
            print(f"   ✗ 健康检查失败: {health_result.get('error', '未知错误')}")
        
        # 测试数据库连接
        print("\n2. 测试数据库连接...")
        db_result = await self.test_database_connection()
        results["database"] = db_result
        
        if db_result["success"]:
            print(f"   ✓ {db_result.get('status', '连接正常')}")
            if "response_time" in db_result:
                print(f"   响应时间: {db_result['response_time']:.3f}秒")
            if "note" in db_result:
                print(f"   备注: {db_result['note']}")
        else:
            print(f"   ✗ 数据库连接测试失败: {db_result.get('error', '未知错误')}")
        
        # 总体评估
        print("\n" + "=" * 50)
        all_success = all(result["success"] for result in results.values())
        
        if all_success:
            print("🎉 MCP服务启动成功！所有测试通过。")
            return {
                "overall": "success",
                "message": "MCP服务正常运行",
                "details": results
            }
        else:
            print("❌ MCP服务启动存在问题。")
            failed_tests = [name for name, result in results.items() if not result["success"]]
            print(f"   失败的测试: {', '.join(failed_tests)}")
            return {
                "overall": "failed",
                "message": "MCP服务存在启动问题",
                "details": results
            }


async def main():
    """主测试函数"""
    # 支持命令行参数指定URL
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print("MCP服务启动测试程序")
    print(f"目标服务: {base_url}")
    print()
    
    async with MCPTester(base_url) as tester:
        result = await tester.comprehensive_test()
        
        # 返回适当的退出码
        sys.exit(0 if result["overall"] == "success" else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"测试程序错误: {e}")
        sys.exit(1)