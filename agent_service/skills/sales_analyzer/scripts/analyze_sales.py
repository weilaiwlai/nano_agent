"""销售数据分析脚本 - 生成趋势图/对比图/占比图。"""

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

    chart_type = data.get("type", "trend")
    title = data.get("title", "销售分析")
    labels = data.get("labels", [])
    values = data.get("values", [])
    extra_values = data.get("extra_values", [])  # 用于对比（如去年同期）
    extra_label = data.get("extra_label", "对比数据")
    y_label = data.get("y_label", "金额（元）")

    fig, ax = plt.subplots(figsize=(12, 6))

    if chart_type == "trend":
        ax.plot(labels, values, marker='o', linewidth=2, color='#4e79a7', label='本期')
        if extra_values:
            ax.plot(labels[:len(extra_values)], extra_values, marker='s', linewidth=2,
                    color='#e15759', linestyle='--', label=extra_label)
            ax.legend()
        ax.set_xlabel("时间", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

    elif chart_type == "bar":
        x = range(len(labels))
        width = 0.35
        ax.bar([i - width/2 for i in x], values, width, label='本期', color='#4e79a7', alpha=0.85)
        if extra_values:
            ax.bar([i + width/2 for i in x][:len(extra_values)], extra_values, width,
                   label=extra_label, color='#e15759', alpha=0.85)
            ax.legend()
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel(y_label, fontsize=12)

    elif chart_type == "pie":
        colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
                  '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']
        explode = [0.05] * len(labels)
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors[:len(labels)], explode=explode[:len(labels)],
            textprops={'fontsize': 10}
        )
        ax.set_title(title, fontsize=14, pad=20)
        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sales_chart_{timestamp}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(json.dumps({"status": "success", "file": filename}, ensure_ascii=False))
        return filename

    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sales_chart_{timestamp}.png"
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
