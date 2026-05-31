"""AI 销售预测 - 数据采集与预测脚本。

用法:
    python predict_sales.py --mode history|forecast|replenish [--granularity day|week|month] [--days 180] [--horizon 30] [--method auto|prophet|moving_avg]

输出: JSON 格式的销售数据和预测结果，供 LLM 进行情景分析。
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

# 尝试导入 Prophet，不可用时降级
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    import pandas as pd
    PROPHET_AVAILABLE = True
except ImportError:
    pass

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


def compute_stats(values: list) -> dict:
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


def get_trunc_sql(granularity: str) -> str:
    """根据粒度返回 SQL DATE_TRUNC 表达式。"""
    if granularity == "week":
        return "DATE_TRUNC('week', order_date)"
    elif granularity == "month":
        return "DATE_TRUNC('month', order_date)"
    else:
        return "DATE(order_date)"


async def collect_history(conn, granularity: str, days: int) -> dict:
    """采集历史销售数据。"""
    today = datetime.now().date()
    start_date = today - timedelta(days=days)
    trunc = get_trunc_sql(granularity)

    # 总体趋势
    trend_rows = await conn.fetch(f"""
        SELECT {trunc} as dt,
               SUM(total_amount) as daily_total,
               COUNT(*) as order_count,
               COUNT(DISTINCT customer_id) as customer_count
        FROM sales_orders
        WHERE order_date >= $1
        GROUP BY {trunc}
        ORDER BY dt
    """, start_date)

    # 分品类趋势
    category_rows = await conn.fetch(f"""
        SELECT {trunc} as dt,
               p.category,
               SUM(so.total_amount) as total_amount,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN products p ON so.product_id = p.product_id
        WHERE so.order_date >= $1
        GROUP BY {trunc}, p.category
        ORDER BY dt, p.category
    """, start_date)

    # 分区域趋势
    region_rows = await conn.fetch(f"""
        SELECT {trunc} as dt,
               c.region,
               SUM(so.total_amount) as total_amount,
               COUNT(*) as order_count
        FROM sales_orders so
        JOIN customers c ON so.customer_id = c.customer_id
        WHERE so.order_date >= $1
        GROUP BY {trunc}, c.region
        ORDER BY dt, c.region
    """, start_date)

    # 构造输出
    daily_amounts = [float(r['daily_total']) for r in trend_rows]

    # 计算趋势方向
    if len(daily_amounts) >= 2:
        first_half = daily_amounts[:len(daily_amounts)//2]
        second_half = daily_amounts[len(daily_amounts)//2:]
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        trend_pct = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
    else:
        trend_pct = 0

    # 分品类组织
    categories = {}
    for r in category_rows:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "date": str(r['dt']),
            "amount": round(float(r['total_amount']), 2),
            "orders": r['order_count'],
        })

    # 分区域组织
    regions = {}
    for r in region_rows:
        reg = r['region']
        if reg not in regions:
            regions[reg] = []
        regions[reg].append({
            "date": str(r['dt']),
            "amount": round(float(r['total_amount']), 2),
            "orders": r['order_count'],
        })

    return {
        "granularity": granularity,
        "period_days": days,
        "start_date": str(start_date),
        "end_date": str(today),
        "trend_direction": "上升" if trend_pct > 2 else ("下降" if trend_pct < -2 else "平稳"),
        "trend_change_pct": round(trend_pct, 1),
        "overall_stats": compute_stats(daily_amounts),
        "daily_trend": [
            {
                "date": str(r['dt']),
                "amount": round(float(r['daily_total']), 2),
                "orders": r['order_count'],
                "customers": r['customer_count'],
            }
            for r in trend_rows
        ],
        "by_category": categories,
        "by_region": regions,
    }


def prophet_forecast(daily_data: list, horizon: int, granularity: str) -> dict:
    """使用 Prophet 进行预测。"""
    if not PROPHET_AVAILABLE:
        return None

    # 准备 Prophet 数据格式
    df_data = []
    for d in daily_data:
        df_data.append({
            "ds": d["date"],
            "y": d["amount"],
        })
    df = pd.DataFrame(df_data)
    df['ds'] = pd.to_datetime(df['ds'])

    # 设置频率
    freq = {"day": "D", "week": "W", "month": "MS"}.get(granularity, "D")

    # 创建并训练模型
    model = Prophet(
        daily_seasonality=(granularity == "day"),
        weekly_seasonality=(granularity == "day"),
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )
    model.fit(df)

    # 生成未来日期并预测
    future = model.make_future_dataframe(periods=horizon, freq=freq)
    forecast = model.predict(future)

    # 提取预测期数据
    forecast_period = forecast.tail(horizon)

    predictions = []
    for _, row in forecast_period.iterrows():
        predictions.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_amount": round(max(0, row['yhat']), 2),
            "lower_bound": round(max(0, row['yhat_lower']), 2),
            "upper_bound": round(max(0, row['yhat_upper']), 2),
        })

    # 趋势分解
    trend_values = forecast_period['trend'].tolist()
    total_predicted = sum(p['predicted_amount'] for p in predictions)

    # 计算精度指标（在历史数据上的拟合）
    historical_fit = forecast.head(len(daily_data))
    actual_vals = [d['amount'] for d in daily_data]
    predicted_vals = historical_fit['yhat'].tolist()
    errors = [abs(a - p) for a, p in zip(actual_vals, predicted_vals) if a > 0]
    mape = (sum(e / a for e, a in zip(errors, actual_vals if len(errors) == len(actual_vals) else [1])) / len(errors) * 100) if errors else None

    return {
        "method": "prophet",
        "horizon": horizon,
        "granularity": granularity,
        "total_predicted": round(total_predicted, 2),
        "daily_average_predicted": round(total_predicted / horizon, 2) if horizon > 0 else 0,
        "confidence_interval_width": round(
            (sum(p['upper_bound'] - p['lower_bound'] for p in predictions) / len(predictions)), 2
        ) if predictions else 0,
        "mape": round(mape, 2) if mape else None,
        "predictions": predictions,
        "trend_summary": {
            "trend_start": round(trend_values[0], 2) if trend_values else 0,
            "trend_end": round(trend_values[-1], 2) if trend_values else 0,
            "trend_direction": "上升" if (trend_values[-1] > trend_values[0]) else "下降",
        },
    }


def moving_average_forecast(daily_data: list, horizon: int, granularity: str) -> dict:
    """使用加权移动平均进行预测（Prophet 不可用时的降级方案）。"""
    amounts = [d['amount'] for d in daily_data]
    if len(amounts) < 7:
        return {"error": "历史数据不足，无法进行预测（至少需要 7 个数据点）"}

    # 计算最近 N 天的加权移动平均（近期权重更高）
    window = min(30, len(amounts))
    recent = amounts[-window:]
    weights = list(range(1, len(recent) + 1))
    wma = sum(a * w for a, w in zip(recent, weights)) / sum(weights)

    # 计算增长率（最近 30 天 vs 前 30 天）
    if len(amounts) >= 60:
        recent_30 = amounts[-30:]
        prev_30 = amounts[-60:-30]
        growth_rate = (sum(recent_30) / len(recent_30)) / (sum(prev_30) / len(prev_30)) - 1
    else:
        growth_rate = 0

    # 检测周周期（如果是按天）
    weekly_factor = [1.0] * 7
    if granularity == "day" and len(amounts) >= 14:
        from collections import defaultdict
        import datetime as dt_module
        dow_amounts = defaultdict(list)
        for d in daily_data[-60:]:
            try:
                dow = dt_module.datetime.strptime(d['date'], '%Y-%m-%d').weekday()
                dow_amounts[dow].append(d['amount'])
            except (ValueError, KeyError):
                pass
        overall_avg = wma
        if overall_avg > 0:
            for dow in range(7):
                if dow in dow_amounts and dow_amounts[dow]:
                    weekly_factor[dow] = (sum(dow_amounts[dow]) / len(dow_amounts[dow])) / overall_avg

    # 计算标准差作为置信区间
    std = (sum((a - wma) ** 2 for a in recent) / len(recent)) ** 0.5

    # 生成预测
    last_date = datetime.strptime(daily_data[-1]['date'], '%Y-%m-%d')
    predictions = []
    total_predicted = 0

    for i in range(1, horizon + 1):
        future_date = last_date + timedelta(days=i)
        # 基础预测 = WMA * (1 + 周增长)
        daily_growth = (1 + growth_rate) ** (1 / 30) - 1
        base_pred = wma * (1 + daily_growth) ** i

        # 应用周周期因子
        if granularity == "day":
            dow = future_date.weekday()
            base_pred *= weekly_factor[dow]

        base_pred = max(0, base_pred)
        lower = max(0, base_pred - 1.96 * std)
        upper = base_pred + 1.96 * std

        predictions.append({
            "date": future_date.strftime('%Y-%m-%d'),
            "predicted_amount": round(base_pred, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
        })
        total_predicted += base_pred

    return {
        "method": "weighted_moving_average",
        "horizon": horizon,
        "granularity": granularity,
        "total_predicted": round(total_predicted, 2),
        "daily_average_predicted": round(total_predicted / horizon, 2) if horizon > 0 else 0,
        "confidence_interval_width": round(2 * 1.96 * std, 2),
        "mape": None,  # WMA 不提供 MAPE
        "wma_base": round(wma, 2),
        "weekly_growth_rate_pct": round(growth_rate * 100, 2),
        "predictions": predictions,
        "trend_summary": {
            "trend_direction": "上升" if growth_rate > 0.02 else ("下降" if growth_rate < -0.02 else "平稳"),
            "daily_growth_rate_pct": round(daily_growth * 100, 3),
        },
        "note": "Prophet 不可用，使用加权移动平均降级方案。建议安装 prophet 以获得更准确的预测。",
    }


async def replenish_suggestions(conn, horizon: int) -> dict:
    """结合预测和库存给出补货建议。"""
    today = datetime.now().date()

    # 获取每个 SKU 的近期日均销量
    daily_sales = await conn.fetch("""
        SELECT p.product_id, p.product_name, p.category,
               COALESCE(AVG(daily.total), 0) as avg_daily_sales,
               COALESCE(STDDEV(daily.total), 0) as std_daily_sales
        FROM products p
        LEFT JOIN (
            SELECT product_id, DATE(order_date) as dt,
                   SUM(quantity) as total
            FROM sales_orders
            WHERE order_date >= $1
            GROUP BY product_id, DATE(order_date)
        ) daily ON p.product_id = daily.product_id
        GROUP BY p.product_id, p.product_name, p.category
    """, today - timedelta(days=30))

    # 获取当前库存
    inventory = await conn.fetch("""
        SELECT product_id, SUM(stock_qty) as total_stock,
               MIN(safety_stock) as safety_stock
        FROM inventory
        GROUP BY product_id
    """)
    inv_map = {r['product_id']: {"stock": r['total_stock'], "safety": r['safety_stock']} for r in inventory}

    suggestions = []
    for r in daily_sales:
        pid = r['product_id']
        avg_daily = float(r['avg_daily_sales'])
        std_daily = float(r['std_daily_sales'])
        inv = inv_map.get(pid, {"stock": 0, "safety": 0})

        forecast_demand = avg_daily * horizon
        safety_buffer = std_daily * (horizon ** 0.5) * 1.65  # 95% 服务水平
        total_need = forecast_demand + safety_buffer + inv['safety']
        suggested_order = max(0, total_need - inv['stock'])

        days_of_stock = inv['stock'] / avg_daily if avg_daily > 0 else 999

        urgency = "低"
        if days_of_stock <= 7:
            urgency = "紧急"
        elif days_of_stock <= 14:
            urgency = "高"
        elif days_of_stock <= 30:
            urgency = "中"

        if suggested_order > 0 or days_of_stock <= horizon:
            suggestions.append({
                "product_id": pid,
                "product_name": r['product_name'],
                "category": r['category'],
                "current_stock": inv['stock'],
                "safety_stock": inv['safety'],
                "avg_daily_sales": round(avg_daily, 2),
                "forecast_demand": round(forecast_demand, 1),
                "suggested_order_qty": round(suggested_order),
                "days_of_stock_remaining": round(days_of_stock, 1),
                "urgency": urgency,
            })

    # 按紧急度排序
    urgency_order = {"紧急": 0, "高": 1, "中": 2, "低": 3}
    suggestions.sort(key=lambda x: (urgency_order.get(x['urgency'], 9), x['days_of_stock_remaining']))

    return {
        "horizon_days": horizon,
        "total_products_analyzed": len(daily_sales),
        "products_need_replenishment": len(suggestions),
        "suggestions": suggestions,
    }


async def main():
    parser = argparse.ArgumentParser(description="AI 销售预测")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["history", "forecast", "replenish"],
                        help="分析模式")
    parser.add_argument("--granularity", type=str, default="day",
                        choices=["day", "week", "month"],
                        help="数据粒度")
    parser.add_argument("--days", type=int, default=180, help="历史回溯天数")
    parser.add_argument("--horizon", type=int, default=30, help="预测天数")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "prophet", "moving_avg"],
                        help="预测方法")
    args = parser.parse_args()

    conn = None
    try:
        conn = await get_connection()

        if args.mode == "history":
            result = await collect_history(conn, args.granularity, args.days)

        elif args.mode == "forecast":
            history = await collect_history(conn, args.granularity, args.days)
            daily_data = history['daily_trend']

            # 选择预测方法
            method = args.method
            if method == "auto":
                method = "prophet" if PROPHET_AVAILABLE else "moving_avg"

            if method == "prophet" and PROPHET_AVAILABLE:
                forecast_result = prophet_forecast(daily_data, args.horizon, args.granularity)
            else:
                forecast_result = moving_average_forecast(daily_data, args.horizon, args.granularity)

            result = {
                "data_type": "sales_forecast",
                "collected_at": datetime.now().isoformat(),
                "prophet_available": PROPHET_AVAILABLE,
                "history_summary": {
                    "period": f"{history['start_date']} ~ {history['end_date']} ({args.days}天)",
                    "trend_direction": history['trend_direction'],
                    "trend_change_pct": history['trend_change_pct'],
                    "avg_daily_sales": history['overall_stats']['mean'],
                },
                "forecast": forecast_result,
            }

        elif args.mode == "replenish":
            result = await replenish_suggestions(conn, args.horizon)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e), "type": type(e).__name__}))
        sys.exit(1)
    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
