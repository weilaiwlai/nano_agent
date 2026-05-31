"""RFM 客户智能分析 - 数据采集与统计脚本。

用法:
    python rfm_analysis.py --mode full|raw|score|detail|trends [--customer-id N] [--r-bins X,Y,Z] [--f-bins X,Y,Z] [--m-bins X,Y,Z]

输出: JSON 格式的客户 RFM 数据，供 LLM 进行智能分群和策略生成。
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
                        if key == 'POSTGRES_USER':
                            DB_CONFIG['user'] = value
                        elif key == 'POSTGRES_PASSWORD':
                            DB_CONFIG['password'] = value
                        elif key == 'POSTGRES_DB':
                            DB_CONFIG['database'] = value
                        elif key == 'POSTGRES_HOST':
                            DB_CONFIG['host'] = value
                        elif key == 'POSTGRES_PORT':
                            DB_CONFIG['port'] = int(value)
            break


load_env()


async def get_connection():
    return await asyncpg.connect(**DB_CONFIG)


def compute_percentiles(values: list, percentiles: list = None) -> dict:
    """计算分位数。"""
    if not values:
        return {}
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in percentiles:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[f"p{p}"] = round(sorted_vals[idx], 2)
    return result


def compute_stats(values: list) -> dict:
    """计算基本统计量 + 分位数。"""
    if not values:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "percentiles": {}}
    n = len(values)
    total = sum(values)
    mean = total / n
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    std = variance ** 0.5
    return {
        "count": n,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "median": round(median, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "percentiles": compute_percentiles(values),
    }


def score_by_bins(value: float, bins: list, reverse: bool = False) -> int:
    """根据分界点打分（1-5）。reverse=True 表示值越小分越高（如 Recency）。"""
    if reverse:
        # 值越小越好（Recency）
        for i, threshold in enumerate(bins):
            if value <= threshold:
                return 5 - i
        return 1
    else:
        # 值越大越好（Frequency, Monetary）
        for i, threshold in enumerate(reversed(bins)):
            if value >= threshold:
                return i + 1
        return 1


async def collect_rfm_raw(conn) -> dict:
    """采集每个客户的 RFM 原始数据。"""
    today = datetime.now().date()

    # RFM 原始数据
    rfm_rows = await conn.fetch("""
        SELECT
            c.customer_id,
            c.customer_name,
            c.region,
            c.level as customer_level,
            MAX(so.order_date) as last_order_date,
            CURRENT_DATE - MAX(so.order_date)::date as recency_days,
            COUNT(DISTINCT so.order_id) as frequency,
            COALESCE(SUM(so.total_amount), 0) as monetary,
            MIN(so.order_date) as first_order_date,
            COALESCE(AVG(so.total_amount), 0) as avg_order_value
        FROM customers c
        LEFT JOIN sales_orders so ON c.customer_id = so.customer_id
        GROUP BY c.customer_id, c.customer_name, c.region, c.level
        ORDER BY monetary DESC
    """)

    customers = []
    recency_list, frequency_list, monetary_list = [], [], []

    for r in rfm_rows:
        recency = int(r['recency_days']) if r['recency_days'] else 999
        frequency = int(r['frequency'])
        monetary = float(r['monetary'])

        customers.append({
            "customer_id": r['customer_id'],
            "customer_name": r['customer_name'],
            "region": r['region'],
            "customer_level": r['customer_level'],
            "recency_days": recency,
            "frequency": frequency,
            "monetary": round(monetary, 2),
            "avg_order_value": round(float(r['avg_order_value']), 2),
            "first_order_date": str(r['first_order_date']) if r['first_order_date'] else None,
            "last_order_date": str(r['last_order_date']) if r['last_order_date'] else None,
        })
        recency_list.append(recency)
        frequency_list.append(frequency)
        monetary_list.append(monetary)

    return {
        "total_customers": len(customers),
        "customers": customers,
        "distributions": {
            "recency_days": compute_stats(recency_list),
            "frequency": compute_stats(frequency_list),
            "monetary": compute_stats(monetary_list),
        },
    }


async def score_and_group(conn, r_bins: list, f_bins: list, m_bins: list) -> dict:
    """根据分界点打分并分群。"""
    rfm_data = await collect_rfm_raw(conn)

    scored_customers = []
    for c in rfm_data['customers']:
        r_score = score_by_bins(c['recency_days'], r_bins, reverse=True)
        f_score = score_by_bins(c['frequency'], f_bins, reverse=False)
        m_score = score_by_bins(c['monetary'], m_bins, reverse=False)

        # 分群逻辑
        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            group = "重要价值客户"
            group_id = 1
        elif r_score >= 4 and f_score >= 4:
            group = "重要发展客户"
            group_id = 2
        elif r_score >= 4 and m_score >= 4:
            group = "重要保持客户"
            group_id = 3
        elif r_score >= 4:
            group = "一般新客户"
            group_id = 4
        elif f_score >= 4 and m_score >= 4:
            group = "重要挽留客户"
            group_id = 5
        elif f_score >= 4:
            group = "一般维护客户"
            group_id = 6
        elif m_score >= 4:
            group = "一般发展客户"
            group_id = 7
        else:
            group = "流失客户"
            group_id = 8

        scored_customers.append({
            **c,
            "r_score": r_score,
            "f_score": f_score,
            "m_score": m_score,
            "rfm_score": f"{r_score}{f_score}{m_score}",
            "rfm_total": r_score + f_score + m_score,
            "group_id": group_id,
            "group_name": group,
        })

    # 统计各群组
    groups = {}
    for c in scored_customers:
        gid = c['group_id']
        if gid not in groups:
            groups[gid] = {
                "group_id": gid,
                "group_name": c['group_name'],
                "count": 0,
                "total_monetary": 0,
                "avg_recency": 0,
                "avg_frequency": 0,
                "customers": [],
            }
        groups[gid]['count'] += 1
        groups[gid]['total_monetary'] += c['monetary']
        groups[gid]['avg_recency'] += c['recency_days']
        groups[gid]['avg_frequency'] += c['frequency']

    # 计算平均值
    for gid, g in groups.items():
        g['avg_recency'] = round(g['avg_recency'] / g['count'], 1)
        g['avg_frequency'] = round(g['avg_frequency'] / g['count'], 1)
        g['total_monetary'] = round(g['total_monetary'], 2)
        g['avg_monetary'] = round(g['total_monetary'] / g['count'], 2)
        # 每个群组保留 Top 5 客户
        group_customers = [c for c in scored_customers if c['group_id'] == gid]
        group_customers.sort(key=lambda x: x['monetary'], reverse=True)
        g['top_customers'] = group_customers[:5]

    # 流失风险客户（R 分 <= 2 且 F/M 分较高）
    at_risk = [
        c for c in scored_customers
        if c['r_score'] <= 2 and (c['f_score'] >= 3 or c['m_score'] >= 3)
    ]
    at_risk.sort(key=lambda x: x['rfm_total'], reverse=True)

    return {
        "scored_at": datetime.now().isoformat(),
        "scoring_criteria": {
            "r_bins": r_bins,
            "f_bins": f_bins,
            "m_bins": m_bins,
            "r_labels": [
                f"≤{r_bins[0]}天=5",
                f"{r_bins[0]+1}-{r_bins[1]}天=4",
                f"{r_bins[1]+1}-{r_bins[2]}天=3",
                f"{r_bins[2]+1}-{r_bins[3]}天=2",
                f">{r_bins[3]}天=1",
            ],
            "f_labels": [
                f"1次=1",
                f"2-{f_bins[0]}次=2",
                f"{f_bins[0]+1}-{f_bins[1]}次=3",
                f"{f_bins[1]+1}-{f_bins[2]}次=4",
                f"≥{f_bins[2]+1}次=5",
            ],
            "m_labels": [
                f"<{m_bins[0]}元=1",
                f"{m_bins[0]}-{m_bins[1]}元=2",
                f"{m_bins[1]+1}-{m_bins[2]}元=3",
                f"{m_bins[2]+1}-{m_bins[3]}元=4",
                f"≥{m_bins[3]+1}元=5",
            ],
        },
        "groups": sorted(groups.values(), key=lambda x: x['group_id']),
        "at_risk_customers": at_risk[:10],
        "total_customers": len(scored_customers),
        "all_customers": scored_customers,
    }


async def customer_detail(conn, customer_id: int) -> dict:
    """单客户详情。"""
    # 基本信息
    customer = await conn.fetchrow("""
        SELECT * FROM customers WHERE customer_id = $1
    """, customer_id)

    if not customer:
        return {"error": f"客户 ID {customer_id} 不存在"}

    # 订单历史
    orders = await conn.fetch("""
        SELECT so.*, p.product_name, p.category
        FROM sales_orders so
        JOIN products p ON so.product_id = p.product_id
        WHERE so.customer_id = $1
        ORDER BY so.order_date DESC
        LIMIT 50
    """, customer_id)

    # 按月统计
    monthly = await conn.fetch("""
        SELECT DATE_TRUNC('month', order_date) as month,
               SUM(total_amount) as total,
               COUNT(*) as order_count
        FROM sales_orders
        WHERE customer_id = $1
        GROUP BY DATE_TRUNC('month', order_date)
        ORDER BY month DESC
    """, customer_id)

    # 品类偏好
    category_pref = await conn.fetch("""
        SELECT p.category,
               SUM(so.total_amount) as total,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN products p ON so.product_id = p.product_id
        WHERE so.customer_id = $1
        GROUP BY p.category
        ORDER BY total DESC
    """, customer_id)

    return {
        "customer_info": {
            "id": customer['customer_id'],
            "name": customer['customer_name'],
            "region": customer['region'],
            "level": customer['level'],
            "contact": customer['contact_phone'],
            "email": customer['contact_email'],
        },
        "order_history": [
            {
                "order_id": r['order_id'],
                "date": str(r['order_date']),
                "product": r['product_name'],
                "category": r['category'],
                "amount": float(r['total_amount']),
                "quantity": r['quantity'],
            }
            for r in orders
        ],
        "monthly_trend": [
            {"month": str(r['month']), "total": float(r['total']), "orders": r['order_count']}
            for r in monthly
        ],
        "category_preference": [
            {"category": r['category'], "total": float(r['total']), "orders": r['order_count']}
            for r in category_pref
        ],
    }


async def rfm_trends(conn) -> dict:
    """RFM 趋势分析：对比近 30 天 vs 前 30 天。"""
    today = datetime.now().date()

    # 最近 30 天的 RFM
    recent_rfm = await conn.fetch("""
        SELECT
            c.customer_id, c.customer_name, c.region,
            CURRENT_DATE - MAX(so.order_date)::date as recency,
            COUNT(DISTINCT so.order_id) as frequency,
            COALESCE(SUM(so.total_amount), 0) as monetary
        FROM customers c
        LEFT JOIN sales_orders so ON c.customer_id = so.customer_id
            AND so.order_date >= $1
        GROUP BY c.customer_id, c.customer_name, c.region
    """, today - timedelta(days=30))

    # 前 30 天的 RFM
    previous_rfm = await conn.fetch("""
        SELECT
            c.customer_id, c.customer_name, c.region,
            CURRENT_DATE - MAX(so.order_date)::date as recency,
            COUNT(DISTINCT so.order_id) as frequency,
            COALESCE(SUM(so.total_amount), 0) as monetary
        FROM customers c
        LEFT JOIN sales_orders so ON c.customer_id = so.customer_id
            AND so.order_date >= $1 AND so.order_date < $2
        GROUP BY c.customer_id, c.customer_name, c.region
    """, today - timedelta(days=60), today - timedelta(days=30))

    prev_map = {r['customer_id']: {"frequency": r['frequency'], "monetary": float(r['monetary'])} for r in previous_rfm}

    trends = []
    for r in recent_rfm:
        cid = r['customer_id']
        prev = prev_map.get(cid, {"frequency": 0, "monetary": 0})
        f_change = r['frequency'] - prev['frequency']
        m_change = float(r['monetary']) - prev['monetary']

        # 流失风险判断
        risk = "低"
        recency = int(r['recency']) if r['recency'] else 999
        if recency > 60 and prev['frequency'] > 0:
            risk = "高"
        elif recency > 30 and prev['frequency'] > 2:
            risk = "中"

        trends.append({
            "customer_id": cid,
            "customer_name": r['customer_name'],
            "region": r['region'],
            "current_recency": recency,
            "current_frequency": r['frequency'],
            "current_monetary": round(float(r['monetary']), 2),
            "previous_frequency": prev['frequency'],
            "previous_monetary": round(prev['monetary'], 2),
            "frequency_change": f_change,
            "monetary_change": round(m_change, 2),
            "churn_risk": risk,
        })

    # 按流失风险排序
    risk_order = {"高": 0, "中": 1, "低": 2}
    trends.sort(key=lambda x: (risk_order[x['churn_risk']], -x['previous_monetary']))

    return {
        "analysis_date": str(today),
        "comparison": "最近30天 vs 前30天",
        "trends": trends,
        "summary": {
            "total_customers": len(trends),
            "high_risk_count": sum(1 for t in trends if t['churn_risk'] == '高'),
            "medium_risk_count": sum(1 for t in trends if t['churn_risk'] == '中'),
            "improving_count": sum(1 for t in trends if t['frequency_change'] > 0),
        },
    }


async def main():
    parser = argparse.ArgumentParser(description="RFM 客户智能分析")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["full", "raw", "score", "detail", "trends"],
                        help="分析模式")
    parser.add_argument("--customer-id", type=int, help="单客户详情模式的客户 ID")
    parser.add_argument("--r-bins", type=str, default="7,30,90,180",
                        help="Recency 分界点（逗号分隔，天数越小分越高）")
    parser.add_argument("--f-bins", type=str, default="2,5,8",
                        help="Frequency 分界点（逗号分隔）")
    parser.add_argument("--m-bins", type=str, default="500,2000,5000,10000",
                        help="Monetary 分界点（逗号分隔）")
    args = parser.parse_args()

    r_bins = [int(x) for x in args.r_bins.split(",")]
    f_bins = [int(x) for x in args.f_bins.split(",")]
    m_bins = [int(x) for x in args.m_bins.split(",")]

    conn = None
    try:
        conn = await get_connection()

        if args.mode == "raw":
            result = await collect_rfm_raw(conn)
        elif args.mode == "score":
            result = await score_and_group(conn, r_bins, f_bins, m_bins)
        elif args.mode == "full":
            raw = await collect_rfm_raw(conn)
            scored = await score_and_group(conn, r_bins, f_bins, m_bins)
            result = {
                "data_type": "rfm_full_analysis",
                "collected_at": datetime.now().isoformat(),
                "raw_data": raw,
                "scored_data": scored,
            }
        elif args.mode == "detail":
            if not args.customer_id:
                print(json.dumps({"error": "--customer-id is required for detail mode"}))
                sys.exit(1)
            result = await customer_detail(conn, args.customer_id)
        elif args.mode == "trends":
            result = await rfm_trends(conn)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}))
        sys.exit(1)
    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
