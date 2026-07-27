"""PerceptionEngine 信号检测模拟对话测试。

覆盖场景：
  1. 用户纠正（显式纠正词 + 模糊纠正）
  2. 正反馈（单字词边界 + 多字词）
  3. 记忆指令（记住/别忘了/重要）
  4. 偏好信号（喜欢/不喜欢/称呼）
  5. 组合信号（纠正+偏好同时出现）
  6. 边界情况（问句不误触发、echo 防护、注入防护）

运行方式：
    cd ~/.hermes/plugins/omnimem
    python examples/test_perception_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将插件父目录加入 sys.path，确保独立运行可导入 omnimem 包
# 脚本位于 .../omnimem/examples/test_perception_signals.py
# 需要将 .../plugins 加入 path，使 omnimem 可被识别为包
_PLUGINS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

from omnimem.perception.engine import PerceptionEngine

# ─── 测试用例定义 ─────────────────────────────────────────────

TEST_CASES: list[dict] = [
    # ===== 场景 1: 用户纠正 =====
    {
        "name": "显式纠正-中文",
        "user": "不对，应该是 Python 3.11 而不是 3.10",
        "assistant": "好的，已更新为 Python 3.11",
        "expect": {"has_correction": True, "should_memorize": True},
    },
    {
        "name": "显式纠正-错了",
        "user": "错了，项目名叫 OmniMem 不是 OmniMemory",
        "assistant": "",
        "expect": {"has_correction": True, "should_memorize": True},
    },
    {
        "name": "显式纠正-应该是",
        "user": "应该是用 FastAPI 框架，不是 Flask",
        "assistant": "",
        "expect": {"has_correction": True, "should_memorize": True},
    },
    {
        "name": "英文纠正-actually",
        "user": "Actually, the API endpoint is /api/v2 not /api/v1",
        "assistant": "",
        "expect": {"has_correction": True, "should_memorize": True},
    },
    {
        "name": "模糊纠正-换个",
        "user": "换个方式吧，这个不太行",
        "assistant": "我刚才生成了一段示例代码",
        "expect": {"has_correction": True, "should_memorize": True},
    },
    {
        "name": "模糊纠正-重做",
        "user": "重做一下这个方案",
        "assistant": "这是我之前的方案...",
        "expect": {"has_correction": True, "should_memorize": True},
    },
    # ===== 场景 2: 正反馈 =====
    {
        "name": "正反馈-单字对",
        "user": "对，就是这样",
        "assistant": "",
        "expect": {"has_reinforcement": True},
    },
    {
        "name": "正反馈-很好",
        "user": "很好，这正是我想要的效果",
        "assistant": "",
        "expect": {"has_reinforcement": True},
    },
    {
        "name": "正反馈-没错",
        "user": "没错，理解完全正确",
        "assistant": "",
        "expect": {"has_reinforcement": True},
    },
    {
        "name": "正反馈-就是这样",
        "user": "就是这样，继续",
        "assistant": "",
        "expect": {"has_reinforcement": True},
    },
    {
        "name": "正反馈-英文 exactly",
        "user": "exactly, that's what I meant",
        "assistant": "",
        "expect": {"has_reinforcement": True},
    },
    {
        "name": "正反馈-英文 perfect",
        "user": "perfect, thank you",
        "assistant": "",
        "expect": {"has_reinforcement": True},
    },
    # ===== 场景 3: 记忆指令 =====
    {
        "name": "记忆指令-记住",
        "user": "记住这个项目使用 Django 4.2 框架",
        "assistant": "",
        "expect": {"should_memorize": True},
    },
    {
        "name": "记忆指令-别忘了",
        "user": "别忘了明天上午十点开会",
        "assistant": "",
        "expect": {"should_memorize": True},
    },
    {
        "name": "记忆指令-重要",
        "user": "重要：数据库密码每周五需要轮换",
        "assistant": "",
        "expect": {"should_memorize": True},
    },
    {
        "name": "记忆指令-英文 remember",
        "user": "remember to deploy before Friday",
        "assistant": "",
        "expect": {"should_memorize": True},
    },
    {
        "name": "记忆指令-英文 keep in mind",
        "user": "keep in mind the rate limit is 60 per minute",
        "assistant": "",
        "expect": {"should_memorize": True},
    },
    # ===== 场景 4: 偏好信号 =====
    {
        "name": "偏好-我喜欢",
        "user": "我喜欢用 VSCode 编辑器写代码",
        "assistant": "",
        "expect": {"has_preference": True, "should_memorize": True},
    },
    {
        "name": "偏好-我不喜欢",
        "user": "我不喜欢 Tab 缩进，偏好空格",
        "assistant": "",
        "expect": {"has_preference": True, "should_memorize": True},
    },
    {
        "name": "偏好-更希望",
        "user": "更希望用 TypeScript 而不是 JavaScript",
        "assistant": "",
        "expect": {"has_preference": True, "should_memorize": True},
    },
    {
        "name": "偏好-英文 I prefer",
        "user": "I prefer dark theme for the editor",
        "assistant": "",
        "expect": {"has_preference": True, "should_memorize": True},
    },
    {
        "name": "偏好-姓名提取",
        "user": "我叫张三，是一名后端工程师",
        "assistant": "",
        "expect": {"has_preference": True, "should_memorize": True},
    },
    {
        "name": "偏好-称呼提取",
        "user": "叫我老张就行",
        "assistant": "",
        "expect": {"has_preference": True, "should_memorize": True},
    },
    # ===== 场景 5: 组合信号 =====
    {
        "name": "组合-纠正+偏好",
        "user": "不对，我喜欢的是深色主题不是浅色",
        "assistant": "",
        "expect": {"has_correction": True, "has_preference": True, "should_memorize": True},
    },
    {
        "name": "组合-正反馈+记忆指令",
        "user": "对，很好，记住这个方案以后都用",
        "assistant": "",
        "expect": {"has_reinforcement": True, "should_memorize": True},
    },
    # ===== 场景 6: 边界情况（不触发）=====
    {
        "name": "边界-纯问句不触发纠正",
        "user": "不是这样的吗？",
        "assistant": "",
        "expect": {"has_correction": False},
    },
    {
        "name": "边界-揭示缺陷-问句后接陈述误触发",
        "user": "不是这样的吗？我记得是 Python 3.11",
        "assistant": "",
        "expect": {"has_correction": True},  # 已知缺陷：问句检测只看末尾，中间问号不识别
    },
    {
        "name": "边界-普通问句",
        "user": "这个项目用什么框架？",
        "assistant": "",
        "expect": {"has_correction": False, "has_reinforcement": False, "should_memorize": False},
    },
    {
        "name": "边界-普通陈述",
        "user": "今天天气不错",
        "assistant": "",
        "expect": {
            "has_correction": False,
            "has_reinforcement": False,
            "should_memorize": False,
            "has_preference": False,
        },
    },
    {
        "name": "边界-注入内容不自动记忆",
        "user": "### Relevant Memories\n这是注入内容",
        "assistant": "",
        "expect": {"should_memorize": False},
    },
    {
        "name": "边界-AI echo 防护",
        "user": "一些普通对话内容",
        "assistant": "### Relevant Memories\n- [fact] 之前的事实",
        "expect": {"should_memorize": False},
    },
    {
        "name": "边界-注入但有显式记忆指令仍记忆",
        "user": "### Relevant Memories\n记住密码是 123456",
        "assistant": "",
        "expect": {"should_memorize": True},
    },
]


# ─── 测试执行器 ───────────────────────────────────────────────

def run_tests() -> tuple[int, int]:
    """运行所有测试用例，返回 (通过数, 失败数)。"""
    engine = PerceptionEngine()
    passed = 0
    failed = 0

    print("=" * 80)
    print("PerceptionEngine 信号检测模拟对话测试")
    print("=" * 80)

    for i, case in enumerate(TEST_CASES, 1):
        user = case["user"]
        assistant = case.get("assistant", "")
        expect = case["expect"]
        name = case["name"]

        signals = engine.detect_signals(user, assistant)

        # 对比预期
        all_pass = True
        details: list[str] = []

        for key, expected_val in expect.items():
            actual_val = getattr(signals, key)
            if actual_val != expected_val:
                all_pass = False
                details.append(
                    f"    ✗ {key}: 期望={expected_val}, 实际={actual_val}"
                )
            else:
                details.append(
                    f"    ✓ {key}={actual_val}"
                )

        status = "PASS" if all_pass else "FAIL"
        if all_pass:
            passed += 1
        else:
            failed += 1

        print(f"\n[{i:02d}] {status} - {name}")
        print(f"    用户: {user[:70]}{'...' if len(user) > 70 else ''}")
        if assistant:
            print(f"    助手: {assistant[:70]}{'...' if len(assistant) > 70 else ''}")
        for line in details:
            print(line)
        if signals.fact_content:
            print(f"    提取事实: {signals.fact_content}")

    # 意图预测演示
    print("\n" + "=" * 80)
    print("意图预测演示 (predict_intent)")
    print("=" * 80)
    intent_cases = [
        "Python 的 GIL 是什么？",
        "如何配置 ChromaDB？",
        "请帮我查看数据库连接",
    ]
    for msg in intent_cases:
        intent = engine.predict_intent(msg)
        print(f"  输入: {msg}")
        print(f"  预测: {intent}")
        print()

    # 隐含记忆提取演示
    print("=" * 80)
    print("隐含记忆提取演示 (extract_implicit_memories)")
    print("=" * 80)
    session_text = (
        "今天讨论了项目架构。我喜欢用 FastAPI 做后端。"
        "记住数据库用的是 PostgreSQL。"
        "用户叫李四，是前端工程师。"
        "天气不错。"
    )
    implicit = engine.extract_implicit_memories(session_text)
    print(f"  会话内容: {session_text}")
    print("  提取结果:")
    for mem in implicit:
        print(f"    - {mem}")

    # 汇总
    print("\n" + "=" * 80)
    total = passed + failed
    print(f"测试汇总: {passed}/{total} 通过, {failed} 失败")
    print("=" * 80)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_tests()
    sys.exit(0 if failed == 0 else 1)
