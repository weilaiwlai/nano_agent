# d:\nano_agent\agent_service\history.py
from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Dict
from datetime import datetime
import json
import logging

import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("nanoagent.agent_service.history")
GRAPH_CHECKPOINTER_POSTGRES_URL = os.getenv("GRAPH_CHECKPOINTER_POSTGRES_URL")

class ConversationHistoryViewer:
    """用于查看和管理对话历史记录的类"""
    
    def __init__(self, postgres_url: str = None):
        self.postgres_url = postgres_url or GRAPH_CHECKPOINTER_POSTGRES_URL
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if not self.postgres_url:
            raise RuntimeError("GRAPH_CHECKPOINTER_POSTGRES_URL 未配置")
        
        # 直接使用 asyncpg 连接 PostgreSQL 数据库
        self.pool = await asyncpg.create_pool(dsn=self.postgres_url)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if hasattr(self, 'pool'):
            await self.pool.close()

    async def list_thread_ids(self) -> List[str]:
        """列出所有线程ID（对话ID）"""
        if not hasattr(self, 'pool'):
            raise RuntimeError("数据库连接未初始化，请使用async with上下文管理器")
            
        # 查询所有不同的thread_id
        query = "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id ASC"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [row['thread_id'] for row in rows]

    async def get_conversation_history(self, thread_id: str, limit: int = 50) -> List[dict]:
        """获取特定线程的对话历史记录"""
        if not hasattr(self, 'pool'):
            raise RuntimeError("数据库连接未初始化，请使用async with上下文管理器")
            
        # 查询指定thread_id的检查点数据，关联checkpoint_blobs表获取实际的检查点数据
        query = """
        SELECT 
            c.thread_id,
            c.checkpoint_id,
            c.parent_checkpoint_id,
            b.blob as checkpoint_data,
            c.checkpoint_ns as created_at
        FROM checkpoints c
        JOIN checkpoint_blobs b ON c.checkpoint_id = b.checkpoint_id
        WHERE c.thread_id = $1 
        ORDER BY c.checkpoint_ns DESC 
        LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, thread_id, limit)
            
            conversations = []
            for row in rows:
                thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_data, created_at = row
                
                # 解析检查点数据中的消息
                messages = []
                try:
                    # 解析checkpoint_data数据（可能是JSON格式）
                    if isinstance(checkpoint_data, str):
                        checkpoint_data = json.loads(checkpoint_data)
                    if isinstance(checkpoint_data, dict):
                        # 检查不同可能的消息存储位置
                        if 'channel_values' in checkpoint_data:
                            channel_values = checkpoint_data['channel_values']
                            if 'messages' in channel_values:
                                for msg in channel_values['messages']:
                                    message_info = self._parse_message(msg)
                                    messages.append(message_info)
                        elif 'messages' in checkpoint_data:
                            # 直接在顶层存储消息
                            for msg in checkpoint_data['messages']:
                                message_info = self._parse_message(msg)
                                messages.append(message_info)
                except Exception as e:
                    logger.warning(f"解析检查点数据失败: {e}")
                
                conversations.append({
                    'thread_id': thread_id,
                    'checkpoint_id': checkpoint_id,
                    'parent_checkpoint_id': parent_checkpoint_id,
                    'created_at': created_at,
                    'messages': messages
                })
            
            return conversations

    def _parse_message(self, msg: dict) -> dict:
        """解析单个消息对象"""
        # 根据消息类型进行分类
        if msg.get('type') == 'human':
            msg_role = 'user'
        elif msg.get('type') == 'ai':
            msg_role = 'assistant'
        elif msg.get('type') == 'system':
            msg_role = 'system'
        elif msg.get('type') == 'tool':
            msg_role = 'tool'
        else:
            msg_role = 'unknown'
            
        content = msg.get('data', {}).get('content', '')
        if not content:
            content = str(msg.get('data', ''))
        
        return {
            'role': msg_role,
            'content': content,
            'timestamp': msg.get('time', ''),
            'type': msg.get('type', 'unknown'),
            'additional_info': {k: v for k, v in msg.items() if k not in ['type', 'data', 'time']}
        }

    async def get_latest_conversation(self, thread_id: str) -> Optional[dict]:
        """获取最新的一次对话"""
        histories = await self.get_conversation_history(thread_id, limit=1)
        return histories[0] if histories else None

    async def get_all_conversations_summary(self) -> List[dict]:
        """获取所有对话的摘要信息"""
        if not hasattr(self, 'pool'):
            raise RuntimeError("数据库连接未初始化，请使用async with上下文管理器")
            
        query = """
        SELECT 
            thread_id,
            COUNT(*) as message_count,
            MAX(checkpoint_ns) as last_activity
        FROM checkpoints 
        GROUP BY thread_id 
        ORDER BY last_activity DESC
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            
            summaries = []
            for row in rows:
                thread_id, message_count, last_activity = row
                summaries.append({
                    'thread_id': thread_id,
                    'message_count': message_count,
                    'last_activity': last_activity
                })
            
            return summaries

# 便捷函数：查看特定对话的历史记录
async def view_conversation_history(thread_id: str, limit: int = 50) -> List[dict]:
    """便捷函数：查看特定对话的历史记录"""
    async with ConversationHistoryViewer() as viewer:
        return await viewer.get_conversation_history(thread_id, limit)

# 便捷函数：列出所有对话
async def list_all_conversations_async() -> List[dict]:
    """便捷函数：列出所有对话"""
    async with ConversationHistoryViewer() as viewer:
        return await viewer.get_all_conversations_summary()

# 便捷函数：列出所有线程ID
async def list_thread_ids_async() -> List[str]:
    """便捷函数：列出所有线程ID"""
    async with ConversationHistoryViewer() as viewer:
        return await viewer.list_thread_ids()

# 同步包装函数
def list_thread_ids() -> List[str]:
    """便捷函数：列出所有线程ID（同步）"""
    return asyncio.run(list_thread_ids_async())

def list_all_conversations() -> List[dict]:
    """便捷函数：列出所有对话（同步）"""
    return asyncio.run(list_all_conversations_async())