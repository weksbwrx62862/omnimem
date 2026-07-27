#!/usr/bin/env python3
"""引用计数与连接关闭日志验证脚本。

模拟多实例创建/关闭场景，验证 temporal_kg.py 和 forgetting.py 的引用计数日志输出。

用法：
    cd /home/xxh/.hermes/plugins/omnimem
    CUDA_VISIBLE_DEVICES="" python3 tests/test_refcount_logs.py
"""

import logging
import re
import sys
import tempfile
from pathlib import Path

# 确保可以从项目根目录导入模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── 日志捕获器 ───────────────────────────────────────────────


class LogCapture(logging.Handler):
    """捕获指定 logger 的日志记录，供后续断言。"""

    def __init__(self, logger_name: str):
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.logger = logging.getLogger(logger_name)
        self.old_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.logger.removeHandler(self)
        self.logger.setLevel(self.old_level)
        super().close()

    def find(self, pattern: str, level: int | None = None) -> list[str]:
        """按正则和可选日志级别筛选消息。"""
        results = []
        for r in self.records:
            if level is not None and r.levelno != level:
                continue
            if re.search(pattern, r.getMessage()):
                results.append(r.getMessage())
        return results

    def clear(self) -> None:
        self.records.clear()


def format_summary(label: str, cap: LogCapture) -> str:
    """格式化捕获的日志为可读摘要。"""
    lines = [f"\n{'='*60}", f"  {label} 日志摘要", f"{'='*60}"]
    for r in cap.records:
        ts = f"{r.levelname:8s}"
        lines.append(f"  [{ts}] {r.getMessage()}")
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


# ─── 测试用例 ────────────────────────────────────────────────


def test_temporal_kg_refcount():
    """测试 TemporalKnowledgeGraph 引用计数日志。"""
    from governance.temporal_kg import (
        TemporalKnowledgeGraph,
        _shared_connections,
    )

    cap = LogCapture("governance.temporal_kg")
    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)

        # ── 场景 1: 首个实例创建 → 应输出 "新建连接" + "引用计数=1"
        tkg1 = TemporalKnowledgeGraph(data_dir)
        msgs = cap.find("新建连接")
        assert msgs, "场景1失败: 未找到'新建连接'日志"
        assert cap.find("引用计数=1"), "场景1失败: 未找到'引用计数=1'"
        conn1_id = id(tkg1._conn)
        print(f"  ✅ 场景1: 首实例创建 → 新建连接, conn_id={conn1_id}, 引用计数=1")
        passed += 1

        # ── 场景 2: 第二个实例创建 → 应输出 "复用共享连接" + "引用计数=2"
        cap.clear()
        tkg2 = TemporalKnowledgeGraph(data_dir)
        msgs = cap.find("复用共享连接")
        assert msgs, "场景2失败: 未找到'复用共享连接'日志"
        assert cap.find("引用计数=2"), "场景2失败: 未找到'引用计数=2'"
        assert id(tkg2._conn) == conn1_id, "场景2失败: 第二实例未复用同一连接"
        print("  ✅ 场景2: 第二实例创建 → 复用共享连接, 引用计数=2")
        passed += 1

        # ── 场景 3: 非最后实例关闭 → 应输出 "非最后实例关闭" + "剩余引用=1"
        cap.clear()
        tkg2.close()
        msgs = cap.find("非最后实例关闭")
        assert msgs, "场景3失败: 未找到'非最后实例关闭'日志"
        assert cap.find("剩余引用=1"), "场景3失败: 未找到'剩余引用=1'"
        # tkg1 的连接应仍可用
        assert tkg1._conn is not None, "场景3失败: tkg1 连接被意外关闭"
        try:
            tkg1._conn.execute("SELECT 1")
        except Exception:
            print("  ❌ 场景3失败: tkg1 连接已死，但应该存活")
            failed += 1
            cap.close()
            return passed, failed
        print("  ✅ 场景3: 非最后实例关闭 → 保留连接, 剩余引用=1, tkg1连接存活")
        passed += 1

        # ── 场景 4: 查询验证 → tkg1 应正常工作
        cap.clear()
        tkg1.add_triple("Alice", "uses", "Python", "2024-01-01")
        tkg1.flush()
        results = tkg1.query_current("Alice", "uses")
        assert len(results) == 1, f"场景4失败: 查询结果数={len(results)}, 期望=1"
        print(f"  ✅ 场景4: tkg2 关闭后 tkg1 查询正常 → 找到 {len(results)} 条记录")
        passed += 1

        # ── 场景 5: 最后实例关闭 → 应输出 "最后一个实例关闭"
        cap.clear()
        tkg1.close()
        msgs = cap.find("最后一个实例关闭")
        assert msgs, "场景5失败: 未找到'最后一个实例关闭'日志"
        assert (
            str(data_dir / "temporal_kg.db") not in _shared_connections
        ), "场景5失败: 共享连接缓存未清理"
        print("  ✅ 场景5: 最后实例关闭 → 真正关闭连接, 缓存已清理")
        passed += 1

        # ── 场景 6: 关闭后重建 → 应输出 "新建连接"（非复用）
        cap.clear()
        tkg3 = TemporalKnowledgeGraph(data_dir)
        msgs = cap.find("新建连接")
        assert msgs, "场景6失败: 未找到'新建连接'日志"
        assert cap.find("引用计数=1"), "场景6失败: 未找到'引用计数=1'"
        tkg3.close()
        print("  ✅ 场景6: 关闭后重建 → 新建连接, 引用计数=1")
        passed += 1

        # ── 场景 7: 多实例并发存活 + 逐个关闭
        cap.clear()
        instances = [TemporalKnowledgeGraph(data_dir) for _ in range(4)]
        # 第一个应是新建连接，后三个应是复用
        new_msgs = cap.find("新建连接")
        reuse_msgs = cap.find("复用共享连接")
        assert len(new_msgs) == 1, f"场景7失败: 新建连接数={len(new_msgs)}, 期望=1"
        assert len(reuse_msgs) == 3, f"场景7失败: 复用连接数={len(reuse_msgs)}, 期望=3"
        assert cap.find("引用计数=4"), "场景7失败: 未找到'引用计数=4'"
        print("  ✅ 场景7a: 4个实例 → 1新建+3复用, 引用计数=4")
        passed += 1

        # 逐个关闭，前3个应 "非最后"，最后1个应 "最后"
        cap.clear()
        for i, inst in enumerate(instances):
            inst.close()
        non_last = cap.find("非最后实例关闭")
        last = cap.find("最后一个实例关闭")
        assert len(non_last) == 3, f"场景7失败: 非最后关闭数={len(non_last)}, 期望=3"
        assert len(last) == 1, f"场景7失败: 最后关闭数={len(last)}, 期望=1"
        print("  ✅ 场景7b: 逐个关闭 → 3个非最后+1个最后")
        passed += 1

    cap.close()
    return passed, failed


def test_forgetting_refcount():
    """测试 ForgettingCurve 引用计数日志。"""
    from governance.forgetting import (
        ForgettingCurve,
    )

    cap = LogCapture("governance.forgetting")
    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as td:
        gov_dir = Path(td)

        # ── 场景 1: 首个实例创建 → 应输出 "新建连接" + "引用计数=1"
        fc1 = ForgettingCurve(gov_dir)
        msgs = cap.find("新建连接")
        assert msgs, "场景1失败: 未找到'新建连接'日志"
        assert cap.find("引用计数=1"), "场景1失败: 未找到'引用计数=1'"
        conn1_id = id(fc1._conn)
        print(f"  ✅ 场景1: 首实例创建 → 新建连接, conn_id={conn1_id}, 引用计数=1")
        passed += 1

        # ── 场景 2: 第二个实例创建 → 应输出 "复用共享连接" + "引用计数=2"
        cap.clear()
        fc2 = ForgettingCurve(gov_dir)
        msgs = cap.find("复用共享连接")
        assert msgs, "场景2失败: 未找到'复用共享连接'日志"
        assert cap.find("引用计数=2"), "场景2失败: 未找到'引用计数=2'"
        print("  ✅ 场景2: 第二实例创建 → 复用共享连接, 引用计数=2")
        passed += 1

        # ── 场景 3: 写入 + 查询验证
        fc1.record_access("mem-001", "fact")
        fc1.flush()
        stage = fc1.get_stage("mem-001")
        assert stage == "active", f"场景3失败: 阶段={stage}, 期望=active"
        print(f"  ✅ 场景3: 写入+查询正常 → stage={stage}")
        passed += 1

        # ── 场景 4: 非最后实例关闭 → 应输出 "非最后实例关闭" + "剩余引用=1"
        cap.clear()
        fc2.close()
        msgs = cap.find("非最后实例关闭")
        assert msgs, "场景4失败: 未找到'非最后实例关闭'日志"
        assert cap.find("剩余引用=1"), "场景4失败: 未找到'剩余引用=1'"
        # fc1 的连接应仍可用
        assert fc1._conn is not None, "场景4失败: fc1 连接被意外关闭"
        stage = fc1.get_stage("mem-001")
        assert stage == "active", f"场景4失败: fc1 查询异常, stage={stage}"
        print("  ✅ 场景4: 非最后实例关闭 → 保留连接, 剩余引用=1, fc1查询正常")
        passed += 1

        # ── 场景 5: 最后实例关闭 → 应输出 "最后一个实例关闭"
        cap.clear()
        fc1.close()
        msgs = cap.find("最后一个实例关闭")
        assert msgs, "场景5失败: 未找到'最后一个实例关闭'日志"
        print("  ✅ 场景5: 最后实例关闭 → 真正关闭连接")
        passed += 1

        # ── 场景 6: 关闭后重建 → 应输出 "新建连接"
        cap.clear()
        fc3 = ForgettingCurve(gov_dir)
        msgs = cap.find("新建连接")
        assert msgs, "场景6失败: 未找到'新建连接'日志"
        assert cap.find("引用计数=1"), "场景6失败: 未找到'引用计数=1'"
        fc3.close()
        print("  ✅ 场景6: 关闭后重建 → 新建连接, 引用计数=1")
        passed += 1

    cap.close()
    return passed, failed


def test_ensure_conn_alive_logging():
    """测试 _ensure_conn_alive 的日志输出。"""
    from governance.temporal_kg import (
        TemporalKnowledgeGraph,
    )

    cap = LogCapture("governance.temporal_kg")
    passed = 0
    failed = 0

    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)

        tkg = TemporalKnowledgeGraph(data_dir)
        tkg.add_triple("Bob", "likes", "Coffee", "2024-01-01")
        tkg.flush()
        cap.clear()

        # ── 场景 1: 手动置 conn=None → 应输出 "连接为 None, 重新初始化"
        old_conn = tkg._conn
        tkg._conn = None
        tkg._ensure_conn_alive()
        msgs = cap.find("连接为 None")
        assert msgs, "场景1失败: 未找到'连接为 None'日志"
        # 验证重建后查询正常
        results = tkg.query_current("Bob", "likes")
        assert len(results) == 1, f"场景1失败: 重建后查询结果数={len(results)}"
        print("  ✅ 场景1: conn=None → 输出'连接为 None', 重建后查询正常")
        passed += 1

        # ── 场景 2: 手动关闭连接 → 应输出 "连接丢失"
        cap.clear()
        conn_obj = tkg._conn
        conn_obj.close()  # 直接关闭底层连接，不经过 close() 方法
        tkg._ensure_conn_alive()
        msgs = cap.find("连接丢失")
        assert msgs, "场景2失败: 未找到'连接丢失'日志"
        # 验证重建后查询正常
        results = tkg.query_current("Bob", "likes")
        assert len(results) == 1, f"场景2失败: 重建后查询结果数={len(results)}"
        print("  ✅ 场景2: 连接被外部关闭 → 输出'连接丢失', 重建后查询正常")
        passed += 1

        tkg.close()

    cap.close()
    return passed, failed


def test_meta_store_fts5_logging():
    """测试 MetaStore FTS5 日志级别降为 debug。"""
    from memory.meta_store import MetaStore

    cap = LogCapture("memory.meta_store")
    passed = 0

    with tempfile.TemporaryDirectory() as td:
        ms = MetaStore(Path(td))
        # FTS5 日志应在 DEBUG 级别
        fts_msgs = cap.find("FTS5", level=logging.DEBUG)
        assert fts_msgs, "FTS5 日志未在 DEBUG 级别输出"
        # 不应有 WARNING 级别的 FTS5 日志
        fts_warn = cap.find("FTS5", level=logging.WARNING)
        assert not fts_warn, f"FTS5 不应有 WARNING 日志，但找到: {fts_warn}"
        # 检查日志包含降级说明
        downgrade_msgs = cap.find("日志级别已从 warning 降为 debug")
        assert downgrade_msgs, "未找到'日志级别已从 warning 降为 debug'说明"
        print("  ✅ MetaStore FTS5: 日志在 DEBUG 级别输出，包含降级说明")
        passed += 1
        ms.close()

    cap.close()
    return passed, 0


# ─── 主流程 ──────────────────────────────────────────────────


def main():
    total_passed = 0
    total_failed = 0

    print("\n" + "=" * 60)
    print("  引用计数与连接关闭日志验证")
    print("=" * 60)

    print("\n[1/4] TemporalKnowledgeGraph 引用计数日志")
    print("-" * 40)
    p, f = test_temporal_kg_refcount()
    total_passed += p
    total_failed += f

    print("\n[2/4] ForgettingCurve 引用计数日志")
    print("-" * 40)
    p, f = test_forgetting_refcount()
    total_passed += p
    total_failed += f

    print("\n[3/4] _ensure_conn_alive 日志")
    print("-" * 40)
    p, f = test_ensure_conn_alive_logging()
    total_passed += p
    total_failed += f

    print("\n[4/4] MetaStore FTS5 日志级别")
    print("-" * 40)
    p, f = test_meta_store_fts5_logging()
    total_passed += p
    total_failed += f

    print("\n" + "=" * 60)
    if total_failed == 0:
        print(f"  ✅ 全部通过: {total_passed} 个场景")
    else:
        print(f"  ❌ 通过: {total_passed}, 失败: {total_failed}")
    print("=" * 60 + "\n")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
