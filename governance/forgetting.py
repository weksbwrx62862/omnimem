"""ForgettingCurve — Ebbinghaus 遗忘曲线驱动的4阶段归档 + 热度分类 + 时间窗口查询。

本模块通过多继承组合三个子模块：
  - forgetting_core: _ForgettingCore — 初始化、数据库连接、自适应阶段配置
  - forgetting_stages: _ForgettingStages — 阶段管理、热度分类、访问记录
  - forgetting_ops: _ForgettingOps — 筛选/FSRS/语义评估、状态查询、flush/close

保持向后兼容：外部代码仍可通过
  from omnimem.governance.forgetting import ForgettingCurve
获取完整类。
"""

from omnimem.governance.forgetting_core import _ForgettingCore, HEAT_LEVELS, STAGES
from omnimem.governance.forgetting_ops import _ForgettingOps
from omnimem.governance.forgetting_stages import _ForgettingStages


class ForgettingCurve(_ForgettingOps, _ForgettingStages, _ForgettingCore):
    """Ebbinghaus 遗忘曲线驱动的4阶段归档 + 热度分类 + 时间窗口查询。

    通过多继承组合 _ForgettingCore（核心/连接）、_ForgettingStages（阶段/热度）、
    _ForgettingOps（筛选/FSRS/状态/工具）三个模块。
    所有公开 API 保持不变。
    """

    pass
