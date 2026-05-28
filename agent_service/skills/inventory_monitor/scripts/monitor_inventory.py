"""库存监控分析脚本 - 生成库存健康图表。"""

import sys
import json
import datetime
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def set_chinese_font():
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def render_chart(data: dict) -> str:
    set_chinese_font()

    chart_type = data.get("type", "health")
    title = data.get("title", "库存分析")
    labels = data.get("labels", [])
    values = data.get("values", [])
    colors_override = data.get("colors", [])

    fig, ax = plt.subplots(figsize=(12, 6))

    if chart_type == "health":
        # 库存健康状态柱状图：绿=正常，橙=低库存，红=缺货
        bar_colors = []
        for v in values:
            if v == 0:
                bar_colors.append('#e15759')  # 缺货-红
            elif isinstance(v, (int, float)) and v < 10:
                bar_colors.append('#f28e2b')  # 低库存-橙
            else:
                bar_colors.append('#59a14f')  # 正常-绿
        if colors_override:
            bar_colors = colors_override
        ax.bar(labels, values, color=bar_colors, alpha=0.85)
        ax.set_ylabel("库存数量", fontsize=12)
        ax.set_xticklabels(labels, rotation=30, ha='right')

    elif chart_type == "pie":
        colors = ['#59a14f', '#f28e2b', '#e15759', '#4e79a7', '#76b7b2',
                  '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']
        explode = [0.05] * len(labels)
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors[:len(labels)], explode=explode[:len(labels)],
            textprops={'fontsize': 10}
        )

    elif chart_type == "bar":
        bar_colors = colors_override if colors_override else ['#4e79a7'] * len(labels)
        ax.barh(labels, values, color=bar_colors, alpha=0.85)
        ax.set_xlabel("数量", fontsize=12)
        ax.invert_yaxis()

    elif chart_type == "turnover":
        # 库存周转天数柱状图，参考线=30天
        bar_colors = ['#e15759' if v > 30 else '#f28e2b' if v > 15 else '#59a14f' for v in values]
        ax.bar(labels, values, color=bar_colors, alpha=0.85)
        ax.axhline(y=30, color='#e15759', linestyle='--', linewidth=1.5, label='警戒线(30天)')
        ax.axhline(y=15, color='#f28e2b', linestyle='--', linewidth=1, label='关注线(15天)')
        ax.legend()
        ax.set_ylabel("周转天数", fontsize=12)
        ax.set_xticklabels(labels, rotation=30, ha='right')

    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y' if chart_type != 'pie' else 'none')
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inventory_chart_{timestamp}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({"status": "success", "file": filename}, ensure_ascii=False))
    return filename


if __name__ == "__main__":
    if len(sys.argv) > 1:
        full_json = " ".join(sys.argv[1:])
        try:
            data = json.loads(full_json)
            render_chart(data)
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "message": f"JSON解析失败: {e}"}, ensure_ascii=False))
    else:
        print(json.dumps({"status": "error", "message": "请提供图表数据 JSON"}, ensure_ascii=False))
