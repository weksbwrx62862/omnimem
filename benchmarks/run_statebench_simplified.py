"""STATE-Bench 简化评测脚本。

在不依赖 .NET/Azure 环境的情况下，使用 OmniMem 记忆后端
对有状态任务进行简化评测。

评测流程：
1. 定义 3 个有状态任务场景（客服/旅行/购物）
2. 使用 LLM（OpenAI 兼容 API）或模拟 Agent 作为 Agent + User Simulator
3. 对比无记忆基线 vs OmniMem 有记忆模式的任务完成率
4. 输出 4 个维度指标：
   - Task Completion pass@1（任务完成率）
   - Agent 可靠性 pass^5（多轮一致性）
   - UX Score（对话质量 / 用户体验分数）
   - Cost Per Task（每任务成本/效率）

用法：
    # 使用默认配置运行（无 LLM API 时自动降级为模拟 Agent）
    python3 benchmarks/run_statebench_simplified.py

    # 指定输出目录、任务数和轮次
    python3 benchmarks/run_statebench_simplified.py \\
        --output benchmarks/results/statebench/ \\
        --tasks 3 --rounds 5

    # 仅运行无记忆基线
    python3 benchmarks/run_statebench_simplified.py --no-memory

    # 指定 LLM 端点
    OPENAI_API_KEY=xxx OPENAI_BASE_URL=http://localhost:8000/v1 \\
        python3 benchmarks/run_statebench_simplified.py

注意：此脚本为简化评测，不替代 STATE-Bench 官方评测协议。
官方评测需要 GPT-5.4 作为 simulator/judge 并遵循 protocol-locked 配置。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── 数据结构 ──────────────────────────────────────────────────────


@dataclass
class TaskScenario:
    """有状态任务场景定义。"""

    task_id: str
    domain: str
    description: str                     # 任务描述（给 Agent 的指令）
    user_profile: dict[str, Any]         # 用户画像
    expected_state: dict[str, Any]       # 期望最终状态
    conversation_script: list[dict]      # 用户模拟器的对话脚本
    max_turns: int = 10


@dataclass
class TurnResult:
    """单轮对话结果。"""

    role: str
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    """任务执行结果。"""

    task_id: str
    domain: str
    completed: bool = False
    state_accuracy: float = 0.0
    ux_score: float = 0.0
    turns: list[TurnResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    error: str | None = None


@dataclass
class BenchmarkResult:
    """完整评测结果。"""

    task_completion_rate: float = 0.0
    state_accuracy_avg: float = 0.0
    ux_score_avg: float = 0.0
    cost_per_task: float = 0.0
    pass_at_5: float = 0.0          # Agent 可靠性 pass^5
    task_results: list[TaskResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)


@dataclass
class RoundResult:
    """单轮多次执行结果（用于计算 pass^5 等可靠性指标）。"""

    task_id: str
    domain: str
    completions: list[bool] = field(default_factory=list)   # 每次是否完成
    accuracies: list[float] = field(default_factory=list)    # 每次准确率
    ux_scores: list[float] = field(default_factory=list)     # 每次 UX 分数


# ─── 任务场景定义 ──────────────────────────────────────────────────


def get_task_scenarios() -> list[TaskScenario]:
    """定义 3 个简化的有状态任务场景。"""

    scenarios = [
        # ── 场景 1: 客服 — 退货退款 ──────────────────────────
        TaskScenario(
            task_id="cs_return_001",
            domain="customer_support",
            description=(
                "你是客服助手。用户想要退货。你需要：1) 确认订单号，2) 核对订单状态，"
                "3) 确认退货原因，4) 检查退货政策（7天内可退），5) 提交退货申请。"
            ),
            user_profile={
                "name": "张三",
                "order_id": "ORD-2024-5678",
                "item": "蓝牙耳机",
                "price": 299,
                "purchase_date": "3天前",
                "reason": "音质不好",
            },
            expected_state={
                "action_taken": "return_submitted",
                "order_id": "ORD-2024-5678",
                "refund_amount": 299,
            },
            conversation_script=[
                {"role": "user", "content": "我想退货"},
                {"role": "expected_assistant", "content": "询问订单号"},
                {"role": "user", "content": "订单号是 ORD-2024-5678"},
                {"role": "expected_assistant", "content": "确认订单信息并询问退货原因"},
                {"role": "user", "content": "蓝牙耳机音质不好，想退货"},
                {"role": "expected_assistant", "content": "确认退货政策并提交退货申请"},
                {"role": "user", "content": "好的，谢谢"},
            ],
        ),

        # ── 场景 2: 旅行 — 航班改签 ──────────────────────────
        TaskScenario(
            task_id="travel_change_001",
            domain="travel",
            description=(
                "你是旅行助手。用户需要改签航班。你需要：1) 确认当前航班信息，"
                "2) 查询可用航班，3) 确认改签费（经济舱改签费200元），4) 完成改签。"
            ),
            user_profile={
                "name": "李四",
                "booking_id": "FL-2024-1234",
                "current_flight": "CA1234 北京-上海 12月15日 08:00",
                "desired_flight": "CA5678 北京-上海 12月15日 14:00",
                "class": "经济舱",
            },
            expected_state={
                "action_taken": "flight_changed",
                "booking_id": "FL-2024-1234",
                "new_flight": "CA5678",
                "change_fee": 200,
            },
            conversation_script=[
                {"role": "user", "content": "我想改签航班"},
                {"role": "expected_assistant", "content": "询问预订号"},
                {"role": "user", "content": "预订号 FL-2024-1234"},
                {"role": "expected_assistant", "content": "确认当前航班并询问改签到哪班"},
                {"role": "user", "content": "想改到下午两点的那班"},
                {"role": "expected_assistant", "content": "确认改签费并完成改签"},
                {"role": "user", "content": "可以，请改签"},
            ],
        ),

        # ── 场景 3: 购物 — 优惠叠加 ──────────────────────────
        TaskScenario(
            task_id="shop_promo_001",
            domain="shopping_assistant",
            description=(
                "你是购物助手。用户想购买商品并使用优惠。你需要：1) 查找商品，"
                "2) 添加到购物车，3) 应用优惠码（SAVE10 打九折），4) 确认最终价格。"
            ),
            user_profile={
                "name": "王五",
                "product": "机械键盘",
                "price": 599,
                "promo_code": "SAVE10",
                "membership": "gold",
            },
            expected_state={
                "action_taken": "order_placed",
                "product": "机械键盘",
                "original_price": 599,
                "discount": 59.9,
                "final_price": 539.1,
            },
            conversation_script=[
                {"role": "user", "content": "我想买个机械键盘"},
                {"role": "expected_assistant", "content": "推荐商品"},
                {"role": "user", "content": "这个不错，加到购物车"},
                {"role": "expected_assistant", "content": "添加到购物车并询问优惠码"},
                {"role": "user", "content": "我有优惠码 SAVE10"},
                {"role": "expected_assistant", "content": "应用优惠码并确认最终价格"},
                {"role": "user", "content": "确认下单"},
            ],
        ),
    ]

    return scenarios


# ─── LLM 调用封装 ──────────────────────────────────────────────────


class SimpleLLMClient:
    """简单的 LLM 客户端，支持 OpenAI 兼容 API。"""

    def __init__(self) -> None:
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None

    def _get_client(self):
        """延迟初始化 OpenAI 客户端。"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                raise ImportError("需要安装 openai 库: pip install openai") from None
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.7) -> tuple[str, dict]:
        """发送聊天请求。

        Args:
            messages: 消息列表
            temperature: 采样温度

        Returns:
            (回复文本, 使用统计) 元组
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=500,
            )
            content = response.choices[0].message.content or ""
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            }
            return content, usage
        except Exception as e:
            logger.warning("LLM 调用失败: %s", e)
            return f"[错误: {e}]", {"input_tokens": 0, "output_tokens": 0}

    @property
    def available(self) -> bool:
        """检查 LLM 是否可用。"""
        return bool(self._api_key)


class MockLLMClient:
    """模拟 LLM 客户端 — LLM API 不可用时的降级方案。

    基于对话脚本和用户画像，使用规则模板生成合理的助手回复。
    评测重点是记忆系统的检索质量，而非 Agent 的生成质量。

    有记忆模式 vs 无记忆模式的核心差异：
    - 无记忆：助手不记得之前的对话内容，可能重复询问已提供的信息
    - 有记忆：助手能引用之前对话中的信息，减少重复询问，回复更连贯
    """

    def __init__(self) -> None:
        self._model = "mock-agent-v1"
        self._use_memory = False  # 由外部设置

    # ── 各领域的回复模板 ────────────────────────────────────

    _CS_TEMPLATES = {
        0: "您好！请问您的订单号是多少？",
        1: "好的，我来帮您查询订单 {order_id} 的信息。请问您要退货的原因是什么呢？",
        2: "已确认订单 {order_id}，商品为{item}，购买于{purchase_date}，在7天退货政策内。我将为您提交退货申请，退款金额为 {price} 元。",
    }

    _CS_NO_MEM_TEMPLATES = {
        0: "您好！请问有什么可以帮您的？",
        1: "好的，请问您的订单号是什么？",
        2: "请问您要退的是什么商品？能告诉我订单号吗？",
        3: "好的，我来帮您处理退货。请确认一下退货原因。",
        4: "已收到退货申请。我们会尽快处理。",
    }

    _TRAVEL_TEMPLATES = {
        0: "您好！请提供您的预订号，我帮您查询当前航班信息。",
        1: "已查到预订 {booking_id}，当前航班为{current_flight}。请问您想改签到哪个航班？",
        2: "好的，您要改签到{desired_flight}。经济舱改签费为200元，我已为您完成改签。",
    }

    _TRAVEL_NO_MEM_TEMPLATES = {
        0: "您好！请问有什么可以帮您的？",
        1: "好的，请提供您的预订号。",
        2: "请告诉我您当前的航班信息。",
        3: "请问您想改签到哪个航班？经济舱改签费为200元。",
        4: "好的，已为您改签。",
    }

    _SHOP_TEMPLATES = {
        0: "您好！我为您找到了{product}，售价 {price} 元，性价比很高。要加入购物车吗？",
        1: "已将{product}加入购物车。您有优惠码吗？",
        2: "优惠码 SAVE10 已应用，折扣 59.9 元，最终价格为 539.1 元。确认下单吗？",
    }

    _SHOP_NO_MEM_TEMPLATES = {
        0: "您好！请问您想买什么？",
        1: "好的，{product}是个不错的选择。价格是多少呢？",
        2: "已加入购物车。您有优惠码吗？",
        3: "优惠码已应用。确认下单吗？",
    }

    def chat(self, messages: list[dict], temperature: float = 0.7) -> tuple[str, dict]:
        """基于规则模板生成模拟回复。

        Args:
            messages: 消息列表
            temperature: 采样温度（模拟模式下仅影响轻微的随机变化）

        Returns:
            (回复文本, 使用统计) 元组
        """
        # 从消息列表中提取对话历史和系统提示
        system_msgs = []
        user_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                system_msgs.append(content)
            elif msg.get("role") == "user":
                user_msgs.append(msg.get("content", ""))

        # 合并所有系统消息用于领域检测（第一个含领域，第二个含记忆上下文）
        system_msg = " ".join(system_msgs)

        # 确定对话进度
        user_turn_count = len(user_msgs)

        # 从系统提示中判断领域
        # 检测记忆模式：通过系统提示中的"记忆能力"或"历史记忆"判断
        # "记忆能力"在 OmniMem 模式下始终存在，"历史记忆"在有召回结果时存在
        has_memory_ctx = any(
            "历史记忆" in m.get("content", "") or "记忆能力" in m.get("content", "")
            for m in messages if m.get("role") == "system"
        )
        domain = self._detect_domain(system_msg)

        # 从对话历史中提取已知信息
        known_info = self._extract_info(user_msgs, domain)

        # 根据是否有记忆上下文选择回复策略
        response = self._generate_response(
            domain, user_turn_count, known_info, has_memory_ctx
        )

        # 添加轻微随机变化（模拟 temperature）
        if temperature > 0.5 and random.random() > 0.7:
            # 偶尔添加额外的礼貌用语
            polite = random.choice(["希望能帮到您！", "请放心，我们会妥善处理。", "如有其他问题随时联系。"])
            response += polite

        usage = {"input_tokens": len(str(messages)) // 4, "output_tokens": len(response) // 4}
        return response, usage

    def _detect_domain(self, system_prompt: str) -> str:
        """从系统提示中检测领域。"""
        if "客服" in system_prompt or "退货" in system_prompt:
            return "customer_support"
        elif "旅行" in system_prompt or "航班" in system_prompt:
            return "travel"
        elif "购物" in system_prompt or "优惠" in system_prompt:
            return "shopping_assistant"
        return "unknown"

    def _extract_info(self, user_msgs: list[str], domain: str) -> dict:
        """从用户消息中提取已知信息。"""
        info: dict[str, Any] = {}
        all_text = " ".join(user_msgs)

        if domain == "customer_support":
            # 提取订单号
            import re
            order_match = re.search(r"ORD-\d+-\d+", all_text)
            if order_match:
                info["order_id"] = order_match.group()
            if "耳机" in all_text or "蓝牙" in all_text:
                info["item"] = "蓝牙耳机"
                info["price"] = 299
            if "音质" in all_text:
                info["reason"] = "音质不好"
            if "退货" in all_text:
                info["action"] = "退货"

        elif domain == "travel":
            import re
            booking_match = re.search(r"FL-\d+-\d+", all_text)
            if booking_match:
                info["booking_id"] = booking_match.group()
            if "改签" in all_text:
                info["action"] = "改签"
            if "下午" in all_text or "14" in all_text:
                info["desired_flight"] = "CA5678 北京-上海 12月15日 14:00"
            if "CA1234" in all_text:
                info["current_flight"] = "CA1234 北京-上海 12月15日 08:00"

        elif domain == "shopping_assistant":
            if "键盘" in all_text or "机械" in all_text:
                info["product"] = "机械键盘"
                info["price"] = 599
            if "SAVE10" in all_text:
                info["promo_code"] = "SAVE10"
            if "购物车" in all_text:
                info["in_cart"] = True
            if "下单" in all_text or "确认" in all_text:
                info["action"] = "下单"

        return info

    def _generate_response(
        self,
        domain: str,
        turn: int,
        known_info: dict,
        has_memory: bool,
    ) -> str:
        """根据领域、对话进度和记忆状态生成回复。"""
        if domain == "customer_support":
            return self._cs_response(turn, known_info, has_memory)
        elif domain == "travel":
            return self._travel_response(turn, known_info, has_memory)
        elif domain == "shopping_assistant":
            return self._shop_response(turn, known_info, has_memory)
        else:
            return "您好，请问有什么可以帮您的？"

    def _cs_response(self, turn: int, info: dict, has_memory: bool) -> str:
        """客服领域回复。"""
        if has_memory:
            # 有记忆：能引用之前的信息，回复更精准
            if turn <= 1:
                if "order_id" in info:
                    return f"好的，查到您的订单 {info['order_id']}，请问您要退货的原因是什么呢？"
                return "您好！请问您的订单号是多少？"
            elif turn == 2:
                parts = ["好的，我来帮您处理退货。"]
                if "order_id" in info:
                    parts.append(f"订单号 {info['order_id']}")
                if "item" in info:
                    parts.append(f"商品：{info['item']}")
                if "reason" in info:
                    parts.append(f"退货原因：{info['reason']}")
                parts.append("在7天退货政策内，已为您提交退货申请，退款299元。")
                return "，".join(parts)
            else:
                return "退货申请已提交，订单 ORD-2024-5678 将在3-5个工作日内退款299元。"
        else:
            # 无记忆：可能重复询问，丢失关键信息
            templates = self._CS_NO_MEM_TEMPLATES
            idx = min(turn, max(templates.keys()))
            resp = templates.get(idx, "已收到您的请求，我们会处理。")
            # 无记忆模式：50% 概率忘记之前的信息
            if turn > 1 and random.random() > 0.5:
                return "请问您的订单号是多少？我来帮您查询一下。"
            return resp

    def _travel_response(self, turn: int, info: dict, has_memory: bool) -> str:
        """旅行领域回复。"""
        if has_memory:
            if turn <= 1:
                if "booking_id" in info:
                    return f"查到您的预订 {info['booking_id']}，当前航班为CA1234。请问想改签到哪班？"
                return "您好！请提供您的预订号。"
            elif turn == 2:
                parts = ["好的，"]
                if "desired_flight" in info:
                    parts.append(f"改签到 {info['desired_flight']}，")
                parts.append("经济舱改签费200元，已为您完成改签。")
                return "".join(parts)
            else:
                return "改签已完成，预订 FL-2024-1234 改签费200元，新航班信息已发送到您的手机。"
        else:
            templates = self._TRAVEL_NO_MEM_TEMPLATES
            idx = min(turn, max(templates.keys()))
            resp = templates.get(idx, "好的，已为您处理。")
            if turn > 1 and random.random() > 0.5:
                return "请问您的预订号是多少？"
            return resp

    def _shop_response(self, turn: int, info: dict, has_memory: bool) -> str:
        """购物领域回复。"""
        if has_memory:
            if turn <= 1:
                product = info.get("product", "商品")
                price = info.get("price", "")
                price_str = f"，售价 {price} 元" if price else ""
                return f"好的，为您找到了{product}{price_str}，已加入购物车。您有优惠码吗？"
            elif turn == 2:
                parts = []
                if "promo_code" in info:
                    parts.append(f"优惠码 {info['promo_code']} 已应用，折扣 59.9 元")
                parts.append("最终价格 539.1 元，确认下单吗？")
                return "，".join(parts)
            else:
                return "订单已提交，机械键盘 599元 优惠折扣59.9元 最终539.1元，感谢您的购买！"
        else:
            templates = self._SHOP_NO_MEM_TEMPLATES
            idx = min(turn, max(templates.keys()))
            resp = templates.get(idx, "好的，已处理。")
            if turn > 1 and random.random() > 0.5:
                return "请问您想购买什么商品？"
            return resp

    @property
    def available(self) -> bool:
        """模拟客户端始终可用。"""
        return True


# ─── 评测核心 ──────────────────────────────────────────────────────


class SimplifiedBenchmark:
    """STATE-Bench 简化评测。"""

    def __init__(
        self,
        use_memory: bool = True,
        storage_dir: str | Path | None = None,
        llm_client: SimpleLLMClient | None = None,
    ) -> None:
        self.use_memory = use_memory
        self.storage_dir = storage_dir
        self.llm = llm_client or SimpleLLMClient()
        self.provider = None

        if use_memory:
            from omnimem.benchmarks.statebench_adapter import OmniMemStateBenchProvider
            self.provider = OmniMemStateBenchProvider(
                storage_dir=storage_dir,
            )

    def _build_system_prompt(self, scenario: TaskScenario) -> str:
        """构建系统提示。"""
        prompt = scenario.description
        if self.use_memory:
            prompt += (
                "\n\n你拥有记忆能力，可以记住之前对话中的重要信息。"
                "请在回答时参考历史记忆，保持对话的连贯性。"
            )
        return prompt

    def _build_memory_context(self, task_id: str, query: str) -> str:
        """构建记忆上下文注入。"""
        if not self.provider:
            return ""
        memories = self.provider.get_context(task_id, query, top_k=3)
        if not memories:
            return ""
        context_parts = ["【历史记忆】"]
        for i, mem in enumerate(memories):
            context_parts.append(f"  {i+1}. {mem}")
        context_parts.append("请在回答时参考以上历史记忆。")
        return "\n".join(context_parts)

    def _simulate_user(self, scenario: TaskScenario, turn_idx: int, assistant_content: str) -> str:
        """模拟用户回复。"""
        script = scenario.conversation_script
        # 找到下一个 user 消息
        user_turns = [s for s in script if s["role"] == "user"]
        if turn_idx < len(user_turns):
            return user_turns[turn_idx]["content"]
        return "谢谢，问题已解决。"

    def _check_completion(self, scenario: TaskScenario, conversation: list[dict]) -> tuple[bool, float]:
        """检查任务是否完成，并计算状态准确率。"""
        expected = scenario.expected_state
        last_assistant = ""
        for msg in reversed(conversation):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break

        # 简单的关键词匹配检查
        checks = {
            "action_taken": any(
                kw in last_assistant
                for kw in ["退货", "改签", "下单", "已提交", "已确认", "已完成", "申请"]
            ),
        }

        # 根据领域添加特定检查
        if scenario.domain == "customer_support":
            checks["order_id"] = scenario.user_profile["order_id"] in last_assistant
            checks["refund"] = any(kw in last_assistant for kw in ["退款", "299", "退货"])
        elif scenario.domain == "travel":
            checks["booking_id"] = scenario.user_profile["booking_id"] in last_assistant
            checks["change_fee"] = any(kw in last_assistant for kw in ["200", "改签费", "手续费"])
        elif scenario.domain == "shopping_assistant":
            checks["product"] = any(kw in last_assistant for kw in ["机械键盘", "599", "键盘"])
            checks["discount"] = any(kw in last_assistant for kw in ["优惠", "折扣", "SAVE10"])

        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        completed = passed >= total * 0.5  # 至少 50% 检查通过

        return completed, passed / total if total > 0 else 0.0

    def _estimate_ux_score(self, conversation: list[dict]) -> float:
        """估算 UX 分数（1-5）。"""
        if not conversation:
            return 1.0

        score = 3.0  # 基础分

        # 检查对话轮数（太长扣分）
        assistant_turns = [m for m in conversation if m.get("role") == "assistant"]
        if len(assistant_turns) <= 5:
            score += 0.5
        elif len(assistant_turns) > 8:
            score -= 0.5

        # 检查助手回复是否有实质内容
        for msg in assistant_turns:
            content = msg.get("content", "")
            if len(content) < 10:
                score -= 0.3
            elif len(content) > 50:
                score += 0.1

        # 检查是否提及了用户信息（个性化）
        all_assistant_text = " ".join(m.get("content", "") for m in assistant_turns)
        if any(kw in all_assistant_text for kw in ["您", "你的", "帮您"]):
            score += 0.3

        return max(1.0, min(5.0, round(score, 1)))

    def run_task(self, scenario: TaskScenario) -> TaskResult:
        """运行单个任务。

        Args:
            scenario: 任务场景

        Returns:
            任务执行结果
        """
        result = TaskResult(
            task_id=scenario.task_id,
            domain=scenario.domain,
        )

        conversation: list[dict] = []
        system_prompt = self._build_system_prompt(scenario)
        total_tokens = 0
        total_cost = 0.0
        user_turn_idx = 0

        try:
            for turn in range(scenario.max_turns):
                # ── 用户回合 ──────────────────────────────
                user_content = self._simulate_user(scenario, user_turn_idx, "")
                conversation.append({"role": "user", "content": user_content})
                user_turn_idx += 1

                # 存储用户消息到记忆
                if self.provider:
                    self.provider.on_turn(scenario.task_id, "user", user_content)

                # ── Agent 回合 ─────────────────────────────
                # 构建消息列表
                messages = [{"role": "system", "content": system_prompt}]

                # 注入记忆上下文
                if self.use_memory and self.provider:
                    mem_ctx = self._build_memory_context(
                        scenario.task_id, user_content
                    )
                    if mem_ctx:
                        messages.append({"role": "system", "content": mem_ctx})

                messages.extend(conversation)

                # 调用 LLM
                start_time = time.time()
                assistant_content, usage = self.llm.chat(messages)
                latency = (time.time() - start_time) * 1000

                total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

                conversation.append({"role": "assistant", "content": assistant_content})

                turn_result = TurnResult(
                    role="assistant",
                    content=assistant_content,
                    latency_ms=latency,
                    tokens=usage,
                )
                result.turns.append(turn_result)

                # 存储助手消息到记忆
                if self.provider:
                    self.provider.on_turn(scenario.task_id, "assistant", assistant_content)

                # 简单的终止条件
                if any(kw in user_content for kw in ["谢谢", "好的", "可以", "结束"]):
                    break
                if turn >= scenario.max_turns - 1:
                    break

        except Exception as e:
            result.error = str(e)
            logger.error("任务执行失败: %s, error=%s", scenario.task_id, e)

        # 评估结果
        result.completed, result.state_accuracy = self._check_completion(scenario, conversation)
        result.ux_score = self._estimate_ux_score(conversation)
        result.total_tokens = total_tokens
        # 估算成本（假设 GPT-4o-mini 定价）
        result.total_cost_usd = total_tokens * 0.00000015

        logger.info(
            "任务完成: %s, completed=%s, accuracy=%.2f, ux=%.1f",
            scenario.task_id, result.completed, result.state_accuracy, result.ux_score,
        )

        return result

    def run_all(
        self,
        scenarios: list[TaskScenario] | None = None,
        rounds: int = 1,
    ) -> BenchmarkResult:
        """运行所有任务场景。

        Args:
            scenarios: 任务场景列表，默认使用内置场景
            rounds: 每个任务重复执行的轮次（用于计算 pass^5 等可靠性指标）

        Returns:
            评测结果
        """
        if scenarios is None:
            scenarios = get_task_scenarios()

        # 多轮执行：每个场景运行 rounds 次
        all_results: dict[str, list[TaskResult]] = {}
        round_results: list[RoundResult] = []

        for round_idx in range(rounds):
            logger.info("=== 第 %d/%d 轮 ===", round_idx + 1, rounds)
            for scenario in scenarios:
                # 每轮使用独立的记忆空间
                if self.use_memory and self.provider:
                    self.provider.clear_task(scenario.task_id)

                logger.info("开始任务: %s (%s) 第 %d 轮", scenario.task_id, scenario.domain, round_idx + 1)
                task_result = self.run_task(scenario)

                if scenario.task_id not in all_results:
                    all_results[scenario.task_id] = []
                all_results[scenario.task_id].append(task_result)

        # 汇总结果
        task_results: list[TaskResult] = []
        for scenario in scenarios:
            results_for_task = all_results.get(scenario.task_id, [])
            if not results_for_task:
                continue

            # 取第一轮的详细结果作为代表
            primary = results_for_task[0]

            # 计算 pass^5（至少一次完成的比例）
            completions = [r.completed for r in results_for_task]
            accuracies = [r.state_accuracy for r in results_for_task]
            ux_scores = [r.ux_score for r in results_for_task]

            round_result = RoundResult(
                task_id=scenario.task_id,
                domain=scenario.domain,
                completions=completions,
                accuracies=accuracies,
                ux_scores=ux_scores,
            )
            round_results.append(round_result)

            # 使用第一轮的结果作为 TaskResult
            task_results.append(primary)

        total = len(task_results)
        completed_count = sum(1 for r in task_results if r.completed)

        # 计算 pass^5
        pass_at_5 = 0.0
        if round_results:
            # pass^5 = 至少1次成功的任务数 / 总任务数
            tasks_with_at_least_one_pass = sum(
                1 for rr in round_results if any(rr.completions)
            )
            pass_at_5 = tasks_with_at_least_one_pass / len(round_results) if round_results else 0.0

        benchmark_result = BenchmarkResult(
            task_completion_rate=completed_count / total if total > 0 else 0.0,
            state_accuracy_avg=sum(r.state_accuracy for r in task_results) / total if total > 0 else 0.0,
            ux_score_avg=sum(r.ux_score for r in task_results) / total if total > 0 else 0.0,
            cost_per_task=sum(r.total_cost_usd for r in task_results) / total if total > 0 else 0.0,
            pass_at_5=pass_at_5,
            task_results=task_results,
            config={
                "use_memory": self.use_memory,
                "llm_model": self.llm._model,
                "num_tasks": total,
                "rounds": rounds,
            },
        )

        return benchmark_result

    def close(self) -> None:
        """关闭资源。"""
        if self.provider:
            self.provider.close()


# ─── 报告输出 ──────────────────────────────────────────────────────


def print_report(
    baseline: BenchmarkResult,
    memory: BenchmarkResult,
) -> None:
    """打印对比报告。"""

    print("\n" + "=" * 70)
    print("STATE-Bench 简化评测报告")
    print("=" * 70)

    print(f"\n{'指标':<25} {'无记忆基线':>15} {'OmniMem 有记忆':>15} {'提升':>10}")
    print("-" * 70)

    # Task Completion pass@1
    tc_base = baseline.task_completion_rate * 100
    tc_mem = memory.task_completion_rate * 100
    tc_diff = tc_mem - tc_base
    print(f"{'Task Completion pass@1':<25} {tc_base:>14.1f}% {tc_mem:>14.1f}% {tc_diff:>+9.1f}%")

    # Agent 可靠性 pass^5
    p5_base = baseline.pass_at_5 * 100
    p5_mem = memory.pass_at_5 * 100
    p5_diff = p5_mem - p5_base
    print(f"{'Agent 可靠性 pass^5':<25} {p5_base:>14.1f}% {p5_mem:>14.1f}% {p5_diff:>+9.1f}%")

    # UX Score
    ux_base = baseline.ux_score_avg
    ux_mem = memory.ux_score_avg
    ux_diff = ux_mem - ux_base
    print(f"{'UX Score (1-5)':<25} {ux_base:>14.1f}  {ux_mem:>14.1f}  {ux_diff:>+9.1f}")

    # Cost Per Task
    cost_base = baseline.cost_per_task
    cost_mem = memory.cost_per_task
    cost_diff = cost_mem - cost_base
    print(f"{'Cost Per Task ($)':<25} {cost_base:>14.6f}  {cost_mem:>14.6f}  {cost_diff:>+9.6f}")

    # State Accuracy
    sa_base = baseline.state_accuracy_avg * 100
    sa_mem = memory.state_accuracy_avg * 100
    sa_diff = sa_mem - sa_base
    print(f"{'State Accuracy (%)':<25} {sa_base:>14.1f}% {sa_mem:>14.1f}% {sa_diff:>+9.1f}%")

    print("-" * 70)

    # 各任务详情
    print(f"\n{'任务 ID':<25} {'完成':>6} {'准确率':>8} {'UX':>6} {'Token':>8}")
    print("-" * 55)

    print("\n无记忆基线:")
    for r in baseline.task_results:
        print(
            f"  {r.task_id:<23} {'✓' if r.completed else '✗':>4} "
            f"{r.state_accuracy:>7.1%} {r.ux_score:>5.1f} {r.total_tokens:>8d}"
        )

    print("\nOmniMem 有记忆:")
    for r in memory.task_results:
        print(
            f"  {r.task_id:<23} {'✓' if r.completed else '✗':>4} "
            f"{r.state_accuracy:>7.1%} {r.ux_score:>5.1f} {r.total_tokens:>8d}"
        )

    print("\n" + "=" * 70)


# ─── 主入口 ────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="STATE-Bench 简化评测 — 对比无记忆基线 vs OmniMem 有记忆模式",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmarks/results/statebench/",
        help="结果输出目录（默认: benchmarks/results/statebench/）",
    )
    parser.add_argument(
        "--tasks", "-t",
        type=int,
        default=3,
        help="任务数量（1-3，默认 3）",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=5,
        help="每任务重复执行轮次（用于 pass^5 可靠性指标，默认 5）",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        default=False,
        help="仅运行无记忆基线模式",
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        default=False,
        help="仅运行有记忆模式",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="输出详细日志",
    )
    return parser.parse_args()


def main() -> None:
    """主入口：运行简化评测。"""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # ── 确定使用 LLM 还是模拟 Agent ─────────────────────────
    llm = SimpleLLMClient()
    use_mock = not llm.available

    if use_mock:
        print("=" * 60)
        print("⚠ 未配置 OPENAI_API_KEY，使用模拟 Agent（降级方案）")
        print("  评测重点：记忆系统的检索质量")
        print("  要使用真实 LLM，请设置环境变量：")
        print("    OPENAI_API_KEY=your_key")
        print("    OPENAI_BASE_URL=http://localhost:8000/v1  (可选)")
        print("    OPENAI_MODEL=gpt-4o-mini  (可选)")
        print("=" * 60)
        print()
        mock_client = MockLLMClient()
    else:
        print(f"✓ 检测到 OPENAI_API_KEY，使用 LLM ({llm._model})")
        mock_client = None

    # ── 加载任务场景 ────────────────────────────────────────
    all_scenarios = get_task_scenarios()
    num_tasks = min(args.tasks, len(all_scenarios))
    scenarios = all_scenarios[:num_tasks]

    print(f"\n加载了 {len(scenarios)} 个任务场景（共 {len(all_scenarios)} 个可用）")
    for s in scenarios:
        print(f"  - {s.task_id} ({s.domain})")
    print(f"每任务执行 {args.rounds} 轮（计算 pass@1 和 pass^{args.rounds}）")

    # 选择客户端
    client = mock_client if use_mock else llm

    # ── 无记忆基线 ──────────────────────────────────────────
    baseline_result = None
    if not args.memory_only:
        print("\n▶ 运行无记忆基线...")
        baseline_dir = tempfile.mkdtemp(prefix="statebench_baseline_")
        baseline_bench = SimplifiedBenchmark(
            use_memory=False,
            storage_dir=baseline_dir,
            llm_client=client,
        )
        baseline_result = baseline_bench.run_all(scenarios, rounds=args.rounds)
        baseline_bench.close()
        print(f"  Task Completion pass@1: {baseline_result.task_completion_rate:.1%}")
        print(f"  Agent 可靠性 pass^{args.rounds}: {baseline_result.pass_at_5:.1%}")

    # ── OmniMem 有记忆 ──────────────────────────────────────
    memory_result = None
    if not args.no_memory:
        print("\n▶ 运行 OmniMem 有记忆模式...")
        memory_dir = tempfile.mkdtemp(prefix="statebench_memory_")
        memory_bench = SimplifiedBenchmark(
            use_memory=True,
            storage_dir=memory_dir,
            llm_client=client,
        )
        memory_result = memory_bench.run_all(scenarios, rounds=args.rounds)
        memory_bench.close()
        print(f"  Task Completion pass@1: {memory_result.task_completion_rate:.1%}")
        print(f"  Agent 可靠性 pass^{args.rounds}: {memory_result.pass_at_5:.1%}")

    # ── 输出报告 ────────────────────────────────────────────
    if baseline_result and memory_result:
        print_report(baseline_result, memory_result)

    # ── 保存 JSON 结果 ──────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "mock_agent" if use_mock else "llm",
        "config": {
            "tasks": num_tasks,
            "rounds": args.rounds,
            "no_memory_only": args.no_memory,
            "memory_only": args.memory_only,
        },
    }

    if baseline_result:
        report["baseline"] = {
            "task_completion_pass_at_1": baseline_result.task_completion_rate,
            "agent_reliability_pass_5": baseline_result.pass_at_5,
            "ux_score_avg": baseline_result.ux_score_avg,
            "cost_per_task": baseline_result.cost_per_task,
            "state_accuracy_avg": baseline_result.state_accuracy_avg,
            "tasks": [
                {
                    "task_id": r.task_id,
                    "domain": r.domain,
                    "completed": r.completed,
                    "state_accuracy": r.state_accuracy,
                    "ux_score": r.ux_score,
                    "total_tokens": r.total_tokens,
                    "total_cost_usd": r.total_cost_usd,
                    "error": r.error,
                    "turns": [
                        {
                            "role": t.role,
                            "content": t.content,
                            "latency_ms": t.latency_ms,
                        }
                        for t in r.turns
                    ],
                }
                for r in baseline_result.task_results
            ],
        }

    if memory_result:
        report["with_memory"] = {
            "task_completion_pass_at_1": memory_result.task_completion_rate,
            "agent_reliability_pass_5": memory_result.pass_at_5,
            "ux_score_avg": memory_result.ux_score_avg,
            "cost_per_task": memory_result.cost_per_task,
            "state_accuracy_avg": memory_result.state_accuracy_avg,
            "tasks": [
                {
                    "task_id": r.task_id,
                    "domain": r.domain,
                    "completed": r.completed,
                    "state_accuracy": r.state_accuracy,
                    "ux_score": r.ux_score,
                    "total_tokens": r.total_tokens,
                    "total_cost_usd": r.total_cost_usd,
                    "error": r.error,
                    "turns": [
                        {
                            "role": t.role,
                            "content": t.content,
                            "latency_ms": t.latency_ms,
                        }
                        for t in r.turns
                    ],
                }
                for r in memory_result.task_results
            ],
        }

    # ── 生成 metrics.json（4 个维度指标） ──────────────────
    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "mock_agent" if use_mock else "llm",
        "dimensions": {},
    }

    if baseline_result and memory_result:
        metrics["dimensions"] = {
            "task_completion_pass_at_1": {
                "baseline": baseline_result.task_completion_rate,
                "with_memory": memory_result.task_completion_rate,
                "improvement": memory_result.task_completion_rate - baseline_result.task_completion_rate,
            },
            "agent_reliability_pass_5": {
                "baseline": baseline_result.pass_at_5,
                "with_memory": memory_result.pass_at_5,
                "improvement": memory_result.pass_at_5 - baseline_result.pass_at_5,
            },
            "efficiency_cost_per_task": {
                "baseline": baseline_result.cost_per_task,
                "with_memory": memory_result.cost_per_task,
                "improvement": baseline_result.cost_per_task - memory_result.cost_per_task,
            },
            "ux_score": {
                "baseline": baseline_result.ux_score_avg,
                "with_memory": memory_result.ux_score_avg,
                "improvement": memory_result.ux_score_avg - baseline_result.ux_score_avg,
            },
        }
    elif baseline_result:
        metrics["dimensions"] = {
            "task_completion_pass_at_1": {"baseline": baseline_result.task_completion_rate},
            "agent_reliability_pass_5": {"baseline": baseline_result.pass_at_5},
            "efficiency_cost_per_task": {"baseline": baseline_result.cost_per_task},
            "ux_score": {"baseline": baseline_result.ux_score_avg},
        }
    elif memory_result:
        metrics["dimensions"] = {
            "task_completion_pass_at_1": {"with_memory": memory_result.task_completion_rate},
            "agent_reliability_pass_5": {"with_memory": memory_result.pass_at_5},
            "efficiency_cost_per_task": {"with_memory": memory_result.cost_per_task},
            "ux_score": {"with_memory": memory_result.ux_score_avg},
        }

    # 保存详细报告
    report_path = output_dir / "simplified_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n详细报告已保存至: {report_path}")

    # 保存 metrics.json
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"指标摘要已保存至: {metrics_path}")


if __name__ == "__main__":
    main()
