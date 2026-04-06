#!/usr/bin/env python3
"""测试Redis连接（带密码认证）"""
 
import redis
import os
from dotenv import load_dotenv
 
# 加载环境变量
load_dotenv()
 
def test_redis():
    """测试Redis连接"""
    try:
        # 获取Redis连接字符串
        redis_url = os.getenv('REDIS_URL', 'redis://202.204.62.144:6379/0')
        print(f"测试连接: {redis_url}")
        
        # 创建Redis连接
        r = redis.from_url(redis_url)
        
        # 测试连接
        result = r.ping()
        print("✅ Redis连接成功")
        
        # 测试基本操作
        r.set('nanoagent_test', '连接测试成功', ex=10)  # 10秒后过期
        value = r.get('nanoagent_test')
        print(f"✅ 基本操作测试通过: {value.decode('utf-8')}")
        
        # 测试列表操作
        r.lpush('nanoagent_test_list', 'test_item')
        list_value = r.lpop('nanoagent_test_list')
        print(f"✅ 列表操作测试通过: {list_value.decode('utf-8')}")
        
        return True
        
    except redis.AuthenticationError:
        print("❌ Redis认证失败：密码错误或需要认证")
        return False
    except redis.ConnectionError as e:
        print(f"❌ Redis连接失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False
 
if __name__ == "__main__":
    print("测试Redis连接...")
    print("-" * 40)
    
    if test_redis():
        print("\n🎉 Redis连接测试通过！")
    else:
        print("\n❌ Redis连接测试失败")
        print("请检查：")
        print("1. Redis服务器是否运行")
        print("2. 密码是否正确")
        print("3. 网络连接是否正常")
        print("4. 防火墙设置")