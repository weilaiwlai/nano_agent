# 创建测试脚本 test_db_connection.py
import asyncpg
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_db_connection():
    try:
        db_url = os.getenv('DB_URL', 'postgresql+asyncpg://postgres:postgres@202.204.62.144:5432/nanoagent')
        # 转换为asyncpg格式
        conn_str = db_url.replace('postgresql+asyncpg://', 'postgresql://')
        conn = await asyncpg.connect(conn_str)
        print("✅ 数据库连接成功")
        await conn.close()
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_db_connection())