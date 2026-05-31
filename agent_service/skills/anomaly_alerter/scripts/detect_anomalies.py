"""智能异常预警 - 数据采集与统计摘要脚本。

用法:
    python detect_anomalies.py [--period 7] [--drill-down sales|inventory|finance] [--region REGION] [--category CATEGORY]

输出: JSON 格式的业务数据摘要，供 LLM 进行异常判断和根因分析。
"""

import sys
import os
import json
import argparse
import asyncio
from datetime import datetime, timedelta

# 添加项目根目录到 path，以便导入数据库模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mcp_server'))

try:
    import asyncpg
except ImportError:
    print(json.dumps({"error": "asyncpg not installed. Run: pip install asyncpg"}))
    sys.exit(1)


# ── 数据库连接 ──
DB_CONFIG = {
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "database": os.getenv("POSTGRES_DB", "nano_agent"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

# ── 从 .env 加载配置 ──
def load_env():
    """尝试从 .env 文件加载数据库配置。"""
    env_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.env'),
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
    """获取数据库连接。"""
    return await asyncpg.connect(**DB_CONFIG)


def compute_stats(values: list) -> dict:
    """计算基本统计量。"""
    if not values:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "total": 0}
    n = len(values)
    total = sum(values)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    std = variance ** 0.5
    return {
        "count": n,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "total": round(total, 2),
    }


async def collect_sales_data(conn, period_days: int) -> dict:
    """采集销售数据：近期 vs 历史同期对比。"""
    today = datetime.now().date()
    period_start = today - timedelta(days=period_days)
    baseline_start = period_start - timedelta(days=period_days)

    # 近期销售（按日聚合）
    recent_rows = await conn.fetch("""
        SELECT DATE(order_date) as dt,
               SUM(total_amount) as daily_total,
               COUNT(*) as order_count
        FROM sales_orders
        WHERE order_date >= $1 AND order_date < $2
        GROUP BY DATE(order_date)
        ORDER BY dt
    """, period_start, today + timedelta(days=1))

    # 历史基线（按日聚合）
    baseline_rows = await conn.fetch("""
        SELECT DATE(order_date) as dt,
               SUM(total_amount) as daily_total,
               COUNT(*) as order_count
        FROM sales_orders
        WHERE order_date >= $1 AND order_date < $2
        GROUP BY DATE(order_date)
        ORDER BY dt
    """, baseline_start, period_start)

    # 近期按区域聚合
    recent_region = await conn.fetch("""
        SELECT c.region,
               SUM(so.total_amount) as total_sales,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN customers c ON so.customer_id = c.customer_id
        WHERE so.order_date >= $1 AND so.order_date < $2
        GROUP BY c.region
        ORDER BY total_sales DESC
    """, period_start, today + timedelta(days=1))

    # 基线按区域聚合
    baseline_region = await conn.fetch("""
        SELECT c.region,
               SUM(so.total_amount) as total_sales,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN customers c ON so.customer_id = c.customer_id
        WHERE so.order_date >= $1 AND so.order_date < $2
        GROUP BY c.region
    """, baseline_start, period_start)

    # 近期按品类聚合
    recent_category = await conn.fetch("""
        SELECT p.category,
               SUM(so.total_amount) as total_sales,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN products p ON so.product_id = p.product_id
        WHERE so.order_date >= $1 AND so.order_date < $2
        GROUP BY p.category
        ORDER BY total_sales DESC
    """, period_start, today + timedelta(days=1))

    # 基线按品类聚合
    baseline_category = await conn.fetch("""
        SELECT p.category,
               SUM(so.total_amount) as total_sales,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN products p ON so.product_id = p.product_id
        WHERE so.order_date >= $1 AND so.order_date < $2
        GROUP BY p.category
    """, baseline_start, period_start)

    # 近期按单品聚合（Top 10 + Bottom 10）
    recent_product = await conn.fetch("""
        SELECT p.product_name, p.category,
               SUM(so.total_amount) as total_sales,
               SUM(so.quantity) as total_qty,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN products p ON so.product_id = p.product_id
        WHERE so.order_date >= $1 AND so.order_date < $2
        GROUP BY p.product_name, p.category
        ORDER BY total_sales DESC
    """, period_start, today + timedelta(days=1))

    # 构造统计数据
    recent_daily_amounts = [float(r['daily_total']) for r in recent_rows]
    baseline_daily_amounts = [float(r['daily_total']) for r in baseline_rows]

    region_baseline_map = {r['region']: float(r['total_sales']) for r in baseline_region}
    category_baseline_map = {r['category']: float(r['total_sales']) for r in baseline_category}

    # 区域对比
    region_comparison = []
    for r in recent_region:
        region = r['region']
        recent_val = float(r['total_sales'])
        baseline_val = region_baseline_map.get(region, 0)
        change_pct = ((recent_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
        region_comparison.append({
            "region": region,
            "recent_sales": round(recent_val, 2),
            "baseline_sales": round(baseline_val, 2),
            "change_pct": round(change_pct, 1),
            "recent_orders": r['order_count'],
        })

    # 品类对比
    category_comparison = []
    for r in recent_category:
        cat = r['category']
        recent_val = float(r['total_sales'])
        baseline_val = category_baseline_map.get(cat, 0)
        change_pct = ((recent_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
        category_comparison.append({
            "category": cat,
            "recent_sales": round(recent_val, 2),
            "baseline_sales": round(baseline_val, 2),
            "change_pct": round(change_pct, 1),
            "recent_orders": r['order_count'],
        })

    # 单品排行
    product_ranking = []
    for r in recent_product[:20]:
        product_ranking.append({
            "name": r['product_name'],
            "category": r['category'],
            "total_sales": round(float(r['total_sales']), 2),
            "total_qty": r['total_qty'],
            "order_count": r['order_count'],
        })

    return {
        "summary": {
            "period_days": period_days,
            "recent_total": round(sum(recent_daily_amounts), 2),
            "baseline_total": round(sum(baseline_daily_amounts), 2),
            "change_pct": round(
                (sum(recent_daily_amounts) - sum(baseline_daily_amounts)) / sum(baseline_daily_amounts) * 100, 1
            ) if baseline_daily_amounts and sum(baseline_daily_amounts) > 0 else 0,
        },
        "daily_stats": {
            "recent": compute_stats(recent_daily_amounts),
            "baseline": compute_stats(baseline_daily_amounts),
        },
        "by_region": region_comparison,
        "by_category": category_comparison,
        "top_products": product_ranking,
    }


async def collect_inventory_data(conn) -> dict:
    """采集库存数据：缺货、滞销、周转率异常。"""
    # 缺货商品
    out_of_stock = await conn.fetch("""
        SELECT p.product_name, p.category, i.warehouse, i.stock_qty, i.safety_stock
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        WHERE i.stock_qty = 0
        ORDER BY p.category, p.product_name
    """)

    # 低库存（低于安全库存）
    low_stock = await conn.fetch("""
        SELECT p.product_name, p.category, i.warehouse, i.stock_qty, i.safety_stock,
               ROUND((i.stock_qty::numeric / NULLIF(i.safety_stock, 0)) * 100, 1) as stock_ratio
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        WHERE i.stock_qty > 0 AND i.stock_qty < i.safety_stock
        ORDER BY (i.stock_qty::numeric / NULLIF(i.safety_stock, 0)) ASC
    """)

    # 库存积压（库存量远超安全库存且近期无销售）
    overstock = await conn.fetch("""
        SELECT p.product_name, p.category, i.warehouse, i.stock_qty, i.safety_stock,
               COALESCE(recent.sales_count, 0) as recent_sales_count
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        LEFT JOIN (
            SELECT product_id, COUNT(*) as sales_count
            FROM sales_orders
            WHERE order_date >= NOW() - INTERVAL '30 days'
            GROUP BY product_id
        ) recent ON p.product_id = recent.product_id
        WHERE i.stock_qty > i.safety_stock * 3
        ORDER BY i.stock_qty DESC
    """)

    # 库存总览
    inventory_summary = await conn.fetchrow("""
        SELECT COUNT(DISTINCT product_id) as total_products,
               SUM(stock_qty) as total_stock,
               SUM(stock_qty) as total_value,
               COUNT(*) FILTER (WHERE stock_qty = 0) as out_of_stock_count,
               COUNT(*) FILTER (WHERE stock_qty > 0 AND stock_qty < safety_stock) as low_stock_count,
               COUNT(*) FILTER (WHERE stock_qty > safety_stock * 3) as overstock_count
        FROM inventory
    """)

    return {
        "summary": {
            "total_products": inventory_summary['total_products'],
            "total_stock": inventory_summary['total_stock'],
            "total_value": round(float(inventory_summary['total_value'] or 0), 2),
            "out_of_stock_count": inventory_summary['out_of_stock_count'],
            "low_stock_count": inventory_summary['low_stock_count'],
            "overstock_count": inventory_summary['overstock_count'],
        },
        "out_of_stock": [
            {"name": r['product_name'], "category": r['category'], "warehouse": r['warehouse'],
             "quantity": r['stock_qty'], "safety_stock": r['safety_stock']}
            for r in out_of_stock
        ],
        "low_stock": [
            {"name": r['product_name'], "category": r['category'], "warehouse": r['warehouse'],
             "quantity": r['stock_qty'], "safety_stock": r['safety_stock'],
             "stock_ratio": float(r['stock_ratio'])}
            for r in low_stock
        ],
        "overstock": [
            {"name": r['product_name'], "category": r['category'], "warehouse": r['warehouse'],
             "quantity": r['stock_qty'], "safety_stock": r['safety_stock'],
             "recent_sales_count": r['recent_sales_count']}
            for r in overstock
        ],
    }


async def collect_finance_data(conn, period_days: int) -> dict:
    """采集财务数据：毛利率波动、费用率异常。"""
    # 月度财务数据
    monthly = await conn.fetch("""
        SELECT year_month, revenue, cogs, gross_profit, opex, net_profit,
               CASE WHEN revenue > 0 THEN ROUND((gross_profit / revenue) * 100, 2) ELSE 0 END as gross_margin,
               CASE WHEN revenue > 0 THEN ROUND((opex / revenue) * 100, 2) ELSE 0 END as expense_ratio,
               CASE WHEN revenue > 0 THEN ROUND((net_profit / revenue) * 100, 2) ELSE 0 END as net_margin
        FROM finance_monthly
        ORDER BY year_month DESC
        LIMIT 12
    """)

    if not monthly:
        return {"summary": "暂无财务数据", "monthly": [], "anomalies": []}

    # 计算各项指标的统计量
    gross_margins = [float(r['gross_margin']) for r in monthly]
    expense_ratios = [float(r['expense_ratio']) for r in monthly]
    net_margins = [float(r['net_margin']) for r in monthly]
    revenues = [float(r['revenue']) for r in monthly]

    # 环比变化
    month_over_month = []
    for i in range(len(monthly) - 1):
        curr = float(monthly[i]['revenue'])
        prev = float(monthly[i + 1]['revenue'])
        change = ((curr - prev) / prev * 100) if prev > 0 else 0
        month_over_month.append({
            "month": str(monthly[i]['year_month']),
            "revenue": round(curr, 2),
            "mom_change_pct": round(change, 1),
        })

    return {
        "summary": {
            "latest_month": str(monthly[0]['year_month']),
            "latest_revenue": round(float(monthly[0]['revenue']), 2),
            "latest_gross_margin": float(monthly[0]['gross_margin']),
            "latest_net_margin": float(monthly[0]['net_margin']),
        },
        "stats": {
            "gross_margin": compute_stats(gross_margins),
            "expense_ratio": compute_stats(expense_ratios),
            "net_margin": compute_stats(net_margins),
            "revenue": compute_stats(revenues),
        },
        "monthly_detail": [
            {
                "month": str(r['year_month']),
                "revenue": round(float(r['revenue']), 2),
                "cost": round(float(r['cogs']), 2),
                "gross_profit": round(float(r['gross_profit']), 2),
                "expenses": round(float(r['opex']), 2),
                "net_profit": round(float(r['net_profit']), 2),
                "gross_margin": float(r['gross_margin']),
                "expense_ratio": float(r['expense_ratio']),
                "net_margin": float(r['net_margin']),
            }
            for r in monthly
        ],
        "month_over_month": month_over_month,
    }


async def drill_down_sales(conn, period_days: int, region: str = None, category: str = None) -> dict:
    """销售下钻分析。"""
    today = datetime.now().date()
    period_start = today - timedelta(days=period_days)

    conditions = ["so.order_date >= $1", "so.order_date < $2"]
    params = [period_start, today + timedelta(days=1)]
    param_idx = 3

    if region:
        conditions.append(f"c.region = ${param_idx}")
        params.append(region)
        param_idx += 1
    if category:
        conditions.append(f"p.category = ${param_idx}")
        params.append(category)
        param_idx += 1

    where = " AND ".join(conditions)

    # 按日趋势
    daily = await conn.fetch(f"""
        SELECT DATE(so.order_date) as dt,
               SUM(so.total_amount) as daily_total,
               COUNT(*) as order_count,
               COUNT(DISTINCT so.customer_id) as customer_count
        FROM sales_orders so
        JOIN customers c ON so.customer_id = c.customer_id
        JOIN products p ON so.product_id = p.product_id
        WHERE {where}
        GROUP BY DATE(so.order_date)
        ORDER BY dt
    """, *params)

    # 按客户
    by_customer = await conn.fetch(f"""
        SELECT c.customer_name, c.region,
               SUM(so.total_amount) as total_amount,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN customers c ON so.customer_id = c.customer_id
        JOIN products p ON so.product_id = p.product_id
        WHERE {where}
        GROUP BY c.customer_name, c.region
        ORDER BY total_amount DESC
        LIMIT 20
    """, *params)

    daily_amounts = [float(r['daily_total']) for r in daily]

    return {
        "filters": {"region": region, "category": category, "period_days": period_days},
        "daily_stats": compute_stats(daily_amounts),
        "daily_trend": [
            {"date": str(r['dt']), "amount": round(float(r['daily_total']), 2),
             "orders": r['order_count'], "customers": r['customer_count']}
            for r in daily
        ],
        "top_customers": [
            {"name": r['customer_name'], "region": r['region'],
             "amount": round(float(r['total_amount']), 2), "orders": r['order_count']}
            for r in by_customer
        ],
    }


async def main():
    parser = argparse.ArgumentParser(description="异常预警数据采集")
    parser.add_argument("--period", type=int, default=7, help="分析周期天数")
    parser.add_argument("--drill-down", type=str, choices=["sales", "inventory", "finance"],
                        help="下钻分析类型")
    parser.add_argument("--region", type=str, help="区域过滤")
    parser.add_argument("--category", type=str, help="品类过滤")
    args = parser.parse_args()

    conn = None
    try:
        conn = await get_connection()

        if args.drill_down == "sales":
            result = await drill_down_sales(conn, args.period, args.region, args.category)
        elif args.drill_down == "inventory":
            result = await collect_inventory_data(conn)
        elif args.drill_down == "finance":
            result = await collect_finance_data(conn, args.period)
        else:
            # 完整数据采集
            result = {
                "data_type": "anomaly_detection",
                "collected_at": datetime.now().isoformat(),
                "sales": await collect_sales_data(conn, args.period),
                "inventory": await collect_inventory_data(conn),
                "finance": await collect_finance_data(conn, args.period),
            }

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}))
        sys.exit(1)
    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
