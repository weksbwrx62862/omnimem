#!/usr/bin/env python3
"""冲突记忆模拟测试脚本。

模拟多轮对话中产生的多种冲突场景，对比 add_only 和 latest 策略的处理差异。

使用方法:
  python3 scripts/mock_conflict_test.py
  python3 scripts/mock_conflict_test.py --strategy add_only
  python3 scripts/mock_conflict_test.py --strategy latest
  python3 scripts/mock_conflict_test.py --verbose   # 显示详细检测过程
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# 确保可以导入 omnimem
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnimem.governance.conflict import ConflictResult, ConflictResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("conflict_test")


# ─── 模拟数据 ──────────────────────────────────────────────────────

MOCK_SCENARIOS = [
    {
        "name": "用户居住地变更",
        "description": "用户三次搬家，测试时序更新冲突",
        "memories": [
            {"id": "addr-v1", "content": "用户住在北京朝阳区", "stored_at": "2024-01-15"},
            {"id": "addr-v2", "content": "用户搬到了上海浦东新区", "stored_at": "2024-06-20"},
            {"id": "addr-v3", "content": "用户现在住在杭州西湖区", "stored_at": "2024-12-01"},
        ],
        "query": "用户现在住在哪里？",
        "expected_answer": "杭州西湖区",
    },
    {
        "name": "技术栈变更",
        "description": "用户从 Python 转向 Go，测试互斥选项冲突",
        "memories": [
            {"id": "tech-v1", "content": "用户主要使用 Python 开发后端服务", "stored_at": "2024-03-10"},
            {"id": "tech-v2", "content": "用户改用 Go 语言开发后端服务", "stored_at": "2024-09-05"},
        ],
        "query": "用户用什么语言开发后端？",
        "expected_answer": "Go 语言",
    },
    {
        "name": "云服务商切换",
        "description": "用户从 AWS 切换到阿里云，测试互斥选项模式",
        "memories": [
            {"id": "cloud-v1", "content": "用户的服务部署在 AWS 上", "stored_at": "2024-02-01"},
            {"id": "cloud-v2", "content": "用户把服务迁移到了阿里云", "stored_at": "2024-08-15"},
        ],
        "query": "用户的服务部署在哪个云上？",
        "expected_answer": "阿里云",
    },
    {
        "name": "纠正错误信息",
        "description": "用户纠正之前的错误偏好，测试否定词冲突",
        "memories": [
            {"id": "corr-v1", "content": "用户喜欢吃辣", "stored_at": "2024-04-01"},
            {"id": "corr-v2", "content": "用户说错了，实际上不喜欢吃辣，偏好清淡", "stored_at": "2024-04-02"},
        ],
        "query": "用户喜欢什么口味？",
        "expected_answer": "清淡（不喜欢辣）",
    },
    {
        "name": "项目状态更新",
        "description": "项目从进行中变为已完成，测试语义更新冲突",
        "memories": [
            {"id": "proj-v1", "content": "Alpha 项目正在进行中，预计下月完成", "stored_at": "2024-05-01"},
            {"id": "proj-v2", "content": "Alpha 项目已经完成了", "stored_at": "2024-06-15"},
        ],
        "query": "Alpha 项目现在什么状态？",
        "expected_answer": "已完成",
    },
    {
        "name": "联系方式变更",
        "description": "手机号变更，测试数字值冲突",
        "memories": [
            {"id": "phone-v1", "content": "用户的手机号是 138-0000-1234", "stored_at": "2024-01-01"},
            {"id": "phone-v2", "content": "用户换了手机号，现在是 139-8888-5678", "stored_at": "2024-07-01"},
        ],
        "query": "用户的手机号是多少？",
        "expected_answer": "139-8888-5678",
    },
    {
        "name": "宠物信息更正",
        "description": "用户把猫的品种从橘猫更正为蓝猫，测试同主题不同实体",
        "memories": [
            {"id": "pet-v1", "content": "用户养了一只橘猫叫小橘", "stored_at": "2024-03-01"},
            {"id": "pet-v2", "content": "用户说之前搞错了，养的是蓝猫不是橘猫，名字还是小橘", "stored_at": "2024-03-02"},
        ],
        "query": "用户养的猫是什么品种？",
        "expected_answer": "蓝猫",
    },
]


# ─── 模拟存储 ──────────────────────────────────────────────────────

class MockMemoryStore:
    """模拟记忆存储，支持写入、标记 superseded、检索过滤。"""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.resolver = ConflictResolver(strategy=strategy)
        self.memories: list[dict] = []
        self.conflict_log: list[dict] = []

    def write(self, memory_id: str, content: str, stored_at: str) -> dict:
        """写入一条记忆，检测与已有记忆的冲突。"""
        # 构造已有记忆列表用于冲突检测
        existing_memories = [
            {"content": m["content"], "memory_id": m["memory_id"]}
            for m in self.memories
        ]

        # 两阶段冲突检测
        conflict = self.resolver.check(content, existing_memories=existing_memories)

        conflict_info = None
        update_marker = None
        if conflict.has_conflict:
            resolution = self.resolver.resolve(content, conflict)
            conflict_info = {
                "conflict_type": conflict.conflict_type,
                "conflicting_with": conflict.existing_id,
                "reason": resolution.reason,
                "is_updated": resolution.is_updated,
                "is_superseded": resolution.is_superseded,
                "superseded_id": resolution.superseded_id,
            }
            if resolution.is_updated:
                update_marker = {
                    "is_updated": True,
                    "is_superseded": True,
                    "superseded_id": resolution.superseded_id,
                }

        # 写入新记忆
        entry = {
            "memory_id": memory_id,
            "content": content,
            "stored_at": stored_at,
            "is_superseded": False,
            "is_updated": update_marker is not None,
            "conflict_info": conflict_info,
        }
        self.memories.append(entry)

        # 处理旧记忆标记
        if update_marker and update_marker["superseded_id"]:
            for m in self.memories:
                if m["memory_id"] == update_marker["superseded_id"]:
                    m["is_superseded"] = True
                    logger.info("  → 旧记忆 %s 被标记为 is_superseded=True", m["memory_id"])

        # 记录冲突日志
        if conflict_info:
            self.conflict_log.append(conflict_info)

        return entry

    def retrieve(self, query: str = "") -> list[dict]:
        """检索记忆，模拟 hybrid_orchestrator 的 superseded 过滤。"""
        # 过滤被 superseded 的记忆
        before = len(self.memories)
        results = [
            r for r in self.memories
            if not r.get("is_superseded")
        ]
        filtered_out = [r["memory_id"] for r in self.memories if r.get("is_superseded")]

        logger.info("  检索: 总记忆 %d 条, 过滤 superseded 后 %d 条 (被过滤: %s)",
                     before, len(results), filtered_out or "无")
        return results

    def summary(self) -> dict:
        """返回存储摘要。"""
        total = len(self.memories)
        superseded = sum(1 for m in self.memories if m.get("is_superseded"))
        retrievable = total - superseded
        conflicts = len(self.conflict_log)
        return {
            "strategy": self.strategy,
            "total_memories": total,
            "superseded": superseded,
            "retrievable": retrievable,
            "conflicts_detected": conflicts,
        }


# ─── 运行测试 ──────────────────────────────────────────────────────

def run_scenario(scenario: dict, strategy: str, verbose: bool = False) -> dict:
    """运行单个场景。"""
    store = MockMemoryStore(strategy=strategy)

    logger.info("=" * 70)
    logger.info("场景: %s", scenario["name"])
    logger.info("说明: %s", scenario["description"])
    logger.info("策略: %s", strategy)
    logger.info("-" * 70)

    # 逐条写入记忆
    for mem in scenario["memories"]:
        logger.info("写入记忆 %s: %s", mem["id"], mem["content"])
        entry = store.write(mem["id"], mem["content"], mem["stored_at"])
        if entry.get("conflict_info"):
            ci = entry["conflict_info"]
            logger.info("  ⚡ 冲突检测: type=%s, conflicts_with=%s, reason=%s",
                        ci["conflict_type"], ci["conflicting_with"], ci["reason"])
            logger.info("  决策: is_updated=%s, is_superseded=%s, superseded_id=%s",
                        ci["is_updated"], ci["is_superseded"], ci["superseded_id"])
        else:
            logger.info("  无冲突, 正常写入")

    # 模拟检索
    logger.info("-" * 70)
    logger.info("查询: %s", scenario["query"])
    results = store.retrieve(scenario["query"])

    # 按时间排序展示检索结果
    by_time = sorted(results, key=lambda x: x["stored_at"], reverse=True)
    logger.info("  检索结果 (按时间倒序):")
    for i, r in enumerate(by_time, 1):
        marker = "★" if i == 1 else " "
        logger.info("  %s [%s] %s (score=N/A, stored_at=%s, is_updated=%s)",
                     marker, r["memory_id"], r["content"],
                     r["stored_at"], r.get("is_updated", False))

    # 判断是否包含正确答案
    top_content = by_time[0]["content"] if by_time else ""
    expected = scenario["expected_answer"]
    has_answer = any(expected[:3] in r["content"] for r in results)

    logger.info("  期望答案: %s", expected)
    logger.info("  答案是否可从检索结果中推断: %s", "是" if has_answer else "否")

    summary = store.summary()
    logger.info("  统计: 总记忆=%d, superseded=%d, 可检索=%d, 冲突次数=%d",
                summary["total_memories"], summary["superseded"],
                summary["retrievable"], summary["conflicts_detected"])

    return {
        "scenario": scenario["name"],
        "strategy": strategy,
        "retrievable_count": summary["retrievable"],
        "superseded_count": summary["superseded"],
        "answer_retrievable": has_answer,
        "retrieved_ids": [r["memory_id"] for r in by_time],
    }


def main():
    parser = argparse.ArgumentParser(description="OmniMem 冲突记忆模拟测试")
    parser.add_argument("--strategy", choices=["add_only", "latest"], default=None,
                        help="仅测试指定策略（默认两者对比）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细检测过程")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("omnimem.governance.conflict").setLevel(logging.DEBUG)

    strategies = [args.strategy] if args.strategy else ["add_only", "latest"]
    all_results = []

    for strategy in strategies:
        logger.info("")
        logger.info("╔════════════════════════════════════════════════════════════════╗")
        logger.info("║  策略: %-54s ║", strategy)
        logger.info("╚════════════════════════════════════════════════════════════════╝")

        for scenario in MOCK_SCENARIOS:
            result = run_scenario(scenario, strategy, verbose=args.verbose)
            all_results.append(result)

    # ─── 汇总对比 ─────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("汇总对比")
    logger.info("=" * 70)
    logger.info("")
    logger.info("%-16s %-12s %-10s %-10s %-12s %s",
                "场景", "策略", "可检索", "superseded", "答案可达", "检索记忆ID")
    logger.info("-" * 70)

    for r in all_results:
        answer_str = "✓" if r["answer_retrievable"] else "✗"
        logger.info("%-16s %-12s %-10d %-10d %-12s %s",
                     r["scenario"], r["strategy"],
                     r["retrievable_count"], r["superseded_count"],
                     answer_str, r["retrieved_ids"])

    # 策略差异总结
    if len(strategies) == 2:
        logger.info("")
        logger.info("策略差异:")
        for scenario in MOCK_SCENARIOS:
            add_only = next(r for r in all_results
                           if r["scenario"] == scenario["name"] and r["strategy"] == "add_only")
            latest = next(r for r in all_results
                         if r["scenario"] == scenario["name"] and r["strategy"] == "latest")
            diff = add_only["retrievable_count"] - latest["retrievable_count"]
            if diff > 0:
                logger.info("  %s: add_only 比 latest 多保留 %d 条记忆",
                            scenario["name"], diff)
            else:
                logger.info("  %s: 两种策略可检索记忆数相同", scenario["name"])


if __name__ == "__main__":
    main()
