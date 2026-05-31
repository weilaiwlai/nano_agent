"""订阅管理脚本。

用法:
    python manage_subscription.py --action create|list|delete|logs --user-id USER_ID [其他参数]

直接操作数据库，通过 HTTP API 的方式在 Skill 内完成订阅管理。
"""

import sys
import os
import json
import argparse
import asyncio
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mcp_server'))

try:
    import asyncpg
except ImportError:
    print(json.dumps({"error": "asyncpg not installed. Run: pip install asyncpg"}))
    sys.exit(1)

DB_CONFIG = {
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "database": os.getenv("POSTGRES_DB", "nano_agent"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}


def load_env():
    env_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'),
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key in ('POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB',
                                   'POSTGRES_HOST', 'POSTGRES_PORT'):
                            DB_CONFIG[key.replace('POSTGRES_', '').lower()] = \
                                int(value) if key == 'POSTGRES_PORT' else value
            break


load_env()

WEEKDAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
               "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}


def _match_field(field: str, value: int, max_val: int) -> bool:
    for part in field.split(','):
        if '/' in part:
            base, step = part.split('/', 1)
            step = int(step)
            if base == '*':
                return value % step == 0
            start = int(base)
            return value >= start and (value - start) % step == 0
        elif '-' in part:
            start, end = part.split('-', 1)
            if int(start) <= value <= int(end):
                return True
        elif part == '*':
            return True
        elif int(part) == value:
            return True
    return False


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        return False
    minute_s, hour_s, dom_s, month_s, dow_s = parts
    if not _match_field(minute_s, dt.minute, 59):
        return False
    if not _match_field(hour_s, dt.hour, 23):
        return False
    if not _match_field(dom_s, dt.day, 31):
        return False
    if not _match_field(month_s, dt.month, 12):
        return False
    if not _match_field(dow_s, dt.weekday(), 6):
        return False
    return True


def compute_next_run(cron_expr: str, after: datetime):
    dt = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(7 * 24 * 60):
        if cron_matches(cron_expr, dt):
            return dt
        dt += timedelta(minutes=1)
    return None


REPORT_TYPE_LABELS = {
    "daily_sales": "每日销售日报",
    "weekly_summary": "经营周报",
    "monthly_finance": "财务月报",
    "anomaly_digest": "异常预警摘要",
}

CRON_LABELS = {
    "0 9 * * *": "每天早上 9:00",
    "30 8 * * *": "每天早上 8:30",
    "0 8 * * *": "每天早上 8:00",
    "0 9 * * 1": "每周一早上 9:00",
    "0 9 * * 5": "每周五早上 9:00",
    "0 17 * * 5": "每周五下午 5:00",
    "0 9 1 * *": "每月 1 号早上 9:00",
    "0 14 15 * *": "每月 15 号下午 2:00",
}


async def create_subscription(conn, args) -> dict:
    """创建订阅。"""
    next_run = compute_next_run(args.cron, datetime.now())
    recipients = args.recipients.split(",") if args.recipients else []
    filters = json.loads(args.filters) if args.filters else {}

    row = await conn.fetchrow("""
        INSERT INTO report_subscriptions
            (user_id, report_type, schedule_cron, recipients, delivery_method, filters, description, next_run_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id, created_at
    """, args.user_id, args.report_type, args.cron,
        recipients, args.delivery, json.dumps(filters, ensure_ascii=False),
        args.description or "", next_run)

    type_label = REPORT_TYPE_LABELS.get(args.report_type, args.report_type)
    cron_label = CRON_LABELS.get(args.cron, args.cron)

    return {
        "status": "success",
        "message": f"✅ 订阅创建成功！\n\n"
                   f"📋 报告类型：{type_label}\n"
                   f"⏰ 定时频率：{cron_label}（{args.cron}）\n"
                   f"📧 接收方式：{args.delivery}\n"
                   f"👥 接收人：{', '.join(recipients) if recipients else '无'}\n"
                   f"📅 下次执行：{next_run.strftime('%Y-%m-%d %H:%M') if next_run else '未计算'}\n"
                   f"🔢 订阅 ID：{row['id']}",
        "subscription_id": row['id'],
        "next_run_at": str(next_run) if next_run else None,
    }


async def list_subscriptions(conn, user_id: str) -> dict:
    """列出用户订阅。"""
    rows = await conn.fetch("""
        SELECT id, report_type, schedule_cron, recipients, delivery_method,
               filters, description, is_active, last_run_at, next_run_at, created_at
        FROM report_subscriptions
        WHERE user_id = $1
        ORDER BY created_at DESC
    """, user_id)

    if not rows:
        return {
            "status": "success",
            "message": "📭 您目前没有任何订阅。\n\n可以对我说「帮我订阅一份每天的销售日报」来创建订阅。",
            "subscriptions": [],
        }

    lines = ["📋 **您的报告订阅列表**\n"]
    for r in rows:
        type_label = REPORT_TYPE_LABELS.get(r['report_type'], r['report_type'])
        status = "🟢 活跃" if r['is_active'] else "🔴 已停用"
        cron_label = CRON_LABELS.get(r['schedule_cron'], r['schedule_cron'])
        recipients = list(r['recipients']) if r['recipients'] else []
        last_run = r['last_run_at'].strftime('%m-%d %H:%M') if r['last_run_at'] else '未执行'
        next_run = r['next_run_at'].strftime('%m-%d %H:%M') if r['next_run_at'] else '未计算'

        lines.append(f"**{r['id']}. {type_label}** {status}")
        lines.append(f"   ⏰ {cron_label}（{r['schedule_cron']}）")
        lines.append(f"   📧 接收人：{', '.join(recipients) if recipients else '无'}")
        lines.append(f"   📅 上次执行：{last_run} | 下次执行：{next_run}")
        if r['description']:
            lines.append(f"   📝 {r['description']}")
        lines.append("")

    return {
        "status": "success",
        "message": "\n".join(lines),
        "subscriptions": [
            {
                "id": r['id'],
                "report_type": r['report_type'],
                "schedule_cron": r['schedule_cron'],
                "is_active": r['is_active'],
                "recipients": list(r['recipients']) if r['recipients'] else [],
                "next_run_at": str(r['next_run_at']) if r['next_run_at'] else None,
            }
            for r in rows
        ],
    }


async def delete_subscription(conn, sub_id: int, user_id: str) -> dict:
    """删除（停用）订阅。"""
    row = await conn.fetchrow("""
        SELECT report_type FROM report_subscriptions
        WHERE id = $1 AND user_id = $2
    """, sub_id, user_id)

    if not row:
        return {"status": "error", "message": f"❌ 未找到 ID 为 {sub_id} 的订阅。"}

    await conn.execute("""
        UPDATE report_subscriptions SET is_active = FALSE, updated_at = NOW()
        WHERE id = $1 AND user_id = $2
    """, sub_id, user_id)

    type_label = REPORT_TYPE_LABELS.get(row['report_type'], row['report_type'])
    return {
        "status": "success",
        "message": f"✅ 已取消「{type_label}」订阅（ID: {sub_id}）。",
    }


async def get_logs(conn, user_id: str) -> dict:
    """获取执行日志。"""
    rows = await conn.fetch("""
        SELECT sl.id, sl.subscription_id, rs.report_type, rs.description,
               sl.status, sl.report_preview, sl.error_message, sl.executed_at
        FROM scheduler_logs sl
        JOIN report_subscriptions rs ON sl.subscription_id = rs.id
        WHERE rs.user_id = $1
        ORDER BY sl.executed_at DESC
        LIMIT 10
    """, user_id)

    if not rows:
        return {
            "status": "success",
            "message": "📭 暂无执行日志。\n\n定时报告会在到达预定时间后自动执行。",
            "logs": [],
        }

    lines = ["📊 **最近的报告执行记录**\n"]
    for r in rows:
        type_label = REPORT_TYPE_LABELS.get(r['report_type'], r['report_type'])
        status_icon = "✅" if r['status'] == 'success' else ("⚠️" if r['status'] == 'skipped' else "❌")
        exec_time = r['executed_at'].strftime('%m-%d %H:%M')
        lines.append(f"{status_icon} **{type_label}** — {exec_time}")
        if r['report_preview']:
            lines.append(f"   📝 {r['report_preview'][:80]}...")
        if r['error_message']:
            lines.append(f"   ❗ {r['error_message'][:80]}")
        lines.append("")

    return {
        "status": "success",
        "message": "\n".join(lines),
        "logs": [
            {
                "log_id": r['id'],
                "report_type": r['report_type'],
                "status": r['status'],
                "executed_at": str(r['executed_at']),
                "preview": r['report_preview'],
            }
            for r in rows
        ],
    }


async def main():
    parser = argparse.ArgumentParser(description="订阅管理")
    parser.add_argument("--action", required=True, choices=["create", "list", "delete", "logs"])
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--report-type", default="daily_sales")
    parser.add_argument("--cron", default="0 9 * * *")
    parser.add_argument("--recipients", default="")
    parser.add_argument("--delivery", default="email")
    parser.add_argument("--filters", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--sub-id", type=int, default=0)
    args = parser.parse_args()

    conn = None
    try:
        conn = await asyncpg.connect(**DB_CONFIG)

        if args.action == "create":
            result = await create_subscription(conn, args)
        elif args.action == "list":
            result = await list_subscriptions(conn, args.user_id)
        elif args.action == "delete":
            result = await delete_subscription(conn, args.sub_id, args.user_id)
        elif args.action == "logs":
            result = await get_logs(conn, args.user_id)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}))
        sys.exit(1)
    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
