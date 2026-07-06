"""
MemoryVisualizer — 记忆系统可视化模块。

提供:
1. 热度分布图
2. 记忆强度分布图
3. 保持率分布图
4. 语义重要性分布图
5. 综合仪表盘
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemoryVisualizer:
    """记忆系统可视化器"""

    def __init__(self, output_dir: Optional[str] = None):
        self._output_dir = output_dir or "/tmp/omnimem_charts"

    def generate_heat_distribution_chart(self, heat_data: dict[str, int]) -> str:
        """生成热度分布图

        Args:
            heat_data: 热度数据 {"hot": N, "warm": N, "neutral": N, "cold": N}

        Returns:
            HTML 图表代码
        """
        labels = ["Hot", "Warm", "Neutral", "Cold"]
        values = [
            heat_data.get("hot", 0),
            heat_data.get("warm", 0),
            heat_data.get("neutral", 0),
            heat_data.get("cold", 0),
        ]
        colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]

        return self._generate_pie_chart("热度分布", labels, values, colors)

    def generate_strength_distribution_chart(self, grade_data: dict[str, int]) -> str:
        """生成记忆强度分布图

        Args:
            grade_data: 等级数据 {"S": N, "A": N, "B": N, "C": N, "D": N}

        Returns:
            HTML 图表代码
        """
        labels = ["S", "A", "B", "C", "D"]
        values = [
            grade_data.get("S", 0),
            grade_data.get("A", 0),
            grade_data.get("B", 0),
            grade_data.get("C", 0),
            grade_data.get("D", 0),
        ]
        colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#95e1d3"]

        return self._generate_bar_chart("记忆强度等级分布", labels, values, colors)

    def generate_retention_distribution_chart(self, retention_data: dict[str, int]) -> str:
        """生成保持率分布图

        Args:
            retention_data: 保持率数据 {"high": N, "medium": N, "low": N}

        Returns:
            HTML 图表代码
        """
        labels = ["高 (>80%)", "中 (50-80%)", "低 (<50%)"]
        values = [
            retention_data.get("high", 0),
            retention_data.get("medium", 0),
            retention_data.get("low", 0),
        ]
        colors = ["#6bcb77", "#ffd93d", "#ff6b6b"]

        return self._generate_pie_chart("保持率分布", labels, values, colors)

    def generate_dashboard(self, stats: dict[str, Any]) -> str:
        """生成综合仪表盘

        Args:
            stats: 统计数据

        Returns:
            HTML 仪表盘代码
        """
        heat_data = stats.get("heat", {})
        grade_data = stats.get("grades", {})
        retention_data = stats.get("retention", {})
        fsrs_stats = stats.get("fsrs", {})

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>OmniMem 记忆系统仪表盘</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            margin: 0;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            font-size: 18px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .stat-label {{
            color: #666;
        }}
        .stat-value {{
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            height: 300px;
            position: relative;
        }}
        .bar {{
            height: 30px;
            margin: 5px 0;
            border-radius: 4px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="card">
            <h2>📊 总览</h2>
            <div class="stat-row">
                <span class="stat-label">总记忆数</span>
                <span class="stat-value">{stats.get('total_memories', 0)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">平均保持率</span>
                <span class="stat-value">{fsrs_stats.get('avg_retention', 0):.1%}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">平均稳定性</span>
                <span class="stat-value">{fsrs_stats.get('avg_stability', 0):.1f} 天</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">升级候选</span>
                <span class="stat-value">{stats.get('upgrade_candidates', 0)}</span>
            </div>
        </div>

        <div class="card">
            <h2>🔥 热度分布</h2>
            {self.generate_heat_distribution_chart(heat_data)}
        </div>

        <div class="card">
            <h2>💪 记忆强度</h2>
            {self.generate_strength_distribution_chart(grade_data)}
        </div>

        <div class="card">
            <h2>📈 保持率分布</h2>
            {self.generate_retention_distribution_chart(retention_data)}
        </div>

        <div class="card">
            <h2>⏰ 最近活动</h2>
            <div class="stat-row">
                <span class="stat-label">24h 内访问</span>
                <span class="stat-value">{stats.get('recent_24h', 0)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">7d 内访问</span>
                <span class="stat-value">{stats.get('recent_7d', 0)}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">30d 内访问</span>
                <span class="stat-value">{stats.get('recent_30d', 0)}</span>
            </div>
        </div>

        <div class="card">
            <h2>🎯 阶段分布</h2>
            {self._generate_stage_bars(stats.get('stages', {}))}
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_stage_bars(self, stages: dict[str, int]) -> str:
        """生成阶段分布条形图"""
        total = sum(stages.values()) or 1
        colors = {
            "active": "#6bcb77",
            "consolidating": "#ffd93d",
            "archived": "#4d96ff",
            "forgotten": "#95e1d3",
        }

        html = ""
        for stage in ["active", "consolidating", "archived", "forgotten"]:
            count = stages.get(stage, 0)
            pct = count / total * 100
            color = colors.get(stage, "#ccc")
            html += f"""
            <div class="bar" style="width: {pct}%; background: {color};">
                {stage}: {count} ({pct:.1f}%)
            </div>
            """

        return html

    def _generate_pie_chart(self, title: str, labels: list, values: list, colors: list) -> str:
        """生成饼图 HTML"""
        total = sum(values) or 1

        html = f'<div style="text-align: center;"><h3>{title}</h3>'
        html += '<div style="display: flex; flex-wrap: wrap; justify-content: center;">'

        for label, value, color in zip(labels, values, colors):
            pct = value / total * 100
            html += f"""
            <div style="margin: 5px; padding: 8px 12px; background: {color}; border-radius: 8px; color: white;">
                {label}: {value} ({pct:.1f}%)
            </div>
            """

        html += "</div></div>"
        return html

    def _generate_bar_chart(self, title: str, labels: list, values: list, colors: list) -> str:
        """生成条形图 HTML"""
        max_val = max(values) if values else 1

        html = f'<div style="text-align: center;"><h3>{title}</h3>'

        for label, value, color in zip(labels, values, colors):
            width = (value / max_val * 100) if max_val > 0 else 0
            html += f"""
            <div class="bar" style="width: {width}%; background: {color};">
                {label}: {value}
            </div>
            """

        html += "</div>"
        return html

    def save_dashboard(
        self,
        stats: dict[str, Any],
        filename: str = "dashboard.html",
        output_dir: Optional[str] = None,
    ) -> str:
        """保存仪表盘到文件

        Args:
            stats: 统计数据
            filename: 文件名
            output_dir: 输出目录

        Returns:
            文件路径
        """
        import os

        target_dir = output_dir or self._output_dir
        os.makedirs(target_dir, exist_ok=True)

        filepath = os.path.join(target_dir, filename)
        html = self.generate_dashboard(stats)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Dashboard saved to %s", filepath)
        return filepath


# 全局实例
_visualizer: Optional[MemoryVisualizer] = None


def get_visualizer(output_dir: Optional[str] = None) -> MemoryVisualizer:
    """获取全局可视化器实例"""
    global _visualizer
    if _visualizer is None:
        _visualizer = MemoryVisualizer(output_dir)
    return _visualizer


def generate_dashboard(stats: dict[str, Any], output_dir: Optional[str] = None) -> str:
    """便捷函数：生成仪表盘"""
    visualizer = get_visualizer(output_dir)
    return visualizer.save_dashboard(stats)
