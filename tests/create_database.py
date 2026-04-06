# create_database.py
import asyncpg
import asyncio

async def create_database():
    try:
        # 先连接到默认的postgres数据库
        conn = await asyncpg.connect(
            host='202.204.62.144',
            port=5432,
            user='postgres',
            password='postgres',
            database='postgres'
        )
        
        # 创建数据库
        await conn.execute('CREATE DATABASE nanoagent')
        print("✅ 数据库 'nanoagent' 创建成功")
        await conn.close()
        
    except Exception as e:
        print(f"❌ 创建数据库失败: {e}")

if __name__ == "__main__":
    asyncio.run(create_database())