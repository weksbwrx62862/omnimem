"""LLM 反思提示词构建与输出解析。

职责：
  - 构建反思用 system/user prompt
  - 调用 _llm_fn / _llm_client 生成观察与心智模型
  - 解析 LLM 输出（中文标记 / JSON）
"""

from __future__ import annotations

import json
import logging
import re

from omnimem.deep.reflect.disposition import Disposition

logger = logging.getLogger(__name__)


def _generate_with_llm(
    self,
    query: str,
    contents: list[str],
    disposition: Disposition,
    max_tokens: int = 800,
) -> tuple[str, str, float] | None:
    """使用 LLM 对记忆内容进行推理归纳。

    Returns:
        (observation, mental_model, confidence) 或 None（LLM 不可用时）
    """
    if self._llm_client is None and self._llm_fn is None:
        logger.warning("ReflectEngine: no LLM client available, skipping LLM call")
        return None

    # 构建推理 prompt
    d = disposition.clamp()
    skepticism_hint = {
        1: "大胆做出结论",
        2: "可以做出初步结论",
        3: "基于现有信息谨慎推理",
        4: "需要更多证据支持，仅给出暂时性判断",
        5: "明确标注不确定性，避免过度推断",
    }.get(d.skepticism, "基于现有信息推理")

    evidence_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(contents[:15]))

    prompt = (
        f"请对以下关于「{query}」的记忆内容进行深度反思和归纳推理。\\n\\n"
        f"推理要求：\\n"
        f"- {skepticism_hint}\\n"
        f"- 从表面事实中提炼深层规律和模式\\n"
        f"- 识别矛盾或不确定性\\n"
        f"- 用简洁的中文表达，避免罗列关键词\\n"
        f"- 心智模型必须是完整的因果陈述句，禁止输出逗号分隔的关键词列表\\n\\n"
        f"记忆内容：\\n{evidence_block}\\n\\n"
        f"请按以下格式输出（严格遵守）：\\n"
        f"【观察】\\n"
        f"（对记忆内容的归纳性总结，2-4句话，提炼核心发现而非复述原文）\\n\\n"
        f"【心智模型】\\n"
        f"（从观察中提炼的规律性认知，1-2个完整的因果陈述句，描述因果关系或模式）\\n\\n"
        f"【置信度】\\n"
        f"（0.0-1.0的数字，表示对上述结论的确信程度）"
    )

    system = (
        "你是一个深度反思引擎。你的任务是从记忆片段中进行归纳推理，"
        "提炼出非平凡的观察和心智模型。"
        "不要简单复述或罗列关键词，要进行真正的推理和抽象。"
        "输出必须严格遵守指定的【观察】【心智模型】【置信度】格式。"
    )

    # ★ R15修复：添加重试机制（最多3次，R17增加到3次以应对截断和关键词堆砌）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw = None
            # ★ R25修复ARCH-1：优先尝试 _llm_fn（经过 provider 的凭证管理），
            # 再尝试直接 _llm_client，确保 LLM 路径被正确调用
            if self._llm_fn is not None:
                try:
                    raw = self._llm_fn(prompt=prompt, system=system, max_tokens=max_tokens)
                except Exception as e:
                    logger.warning(
                        "ReflectEngine _llm_fn failed (attempt %d/%d): %s: %s",
                        attempt + 1, max_retries, type(e).__name__, e,
                    )
            if not raw and self._llm_client is not None:
                try:
                    result = self._llm_client.call_sync(
                        prompt=prompt, system=system, max_tokens=max_tokens, temperature=0.5,
                    )
                    raw = result.content if result else None
                except Exception as e:
                    logger.warning(
                        "ReflectEngine _llm_client failed (attempt %d/%d): %s: %s",
                        attempt + 1, max_retries, type(e).__name__, e,
                    )
            if not raw or not raw.strip():
                if attempt < max_retries - 1:
                    logger.warning(
                        "ReflectEngine LLM returned empty (attempt %d/%d), retrying...",
                        attempt + 1,
                        max_retries,
                    )
                    continue
                logger.warning("LLM generation failed, degrading to rule-based")
                return None

            # ★ R17修复：检测截断 — 如果输出没有结束标记（【置信度】），可能被截断
            has_complete_structure = bool(re.search(r"【置信度】|置信度[：:]", raw))
            if not has_complete_structure and attempt < max_retries - 1:
                logger.warning(
                    "ReflectEngine LLM output appears truncated (attempt %d/%d), retrying...",
                    attempt + 1,
                    max_retries,
                )
                # 截断重试时增加max_tokens
                max_tokens = min(max_tokens + 200, 1200)
                continue

            # 解析 LLM 输出
            obs, model, conf = self._parse_llm_output(raw.strip())

            # ★ R17修复：对截断的observation进行修补
            if obs and len(obs) < 15 and attempt < max_retries - 1:
                logger.warning(
                    "ReflectEngine LLM observation too short (%d chars), likely truncated (attempt %d/%d)",
                    len(obs),
                    attempt + 1,
                    max_retries,
                )
                continue

            # 验证mental_model质量
            if model and self._is_keyword_stuffing(model):
                if attempt < max_retries - 1:
                    logger.warning(
                        "ReflectEngine LLM returned keyword stuffing (attempt %d/%d), retrying...",
                        attempt + 1,
                        max_retries,
                    )
                    # 关键词堆砌重试时，在prompt中增加更明确的反堆砌指令
                    if attempt == max_retries - 2:
                        prompt += (
                            "\n\n★ 重要提醒：你的【心智模型】输出必须是完整的自然语言句子，"
                            "描述因果关系或模式规律，严禁输出关键词列表！"
                        )
                    continue

            return (obs, model, conf)

        except Exception as e:
            logger.warning(
                "ReflectEngine LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e
            )
            if attempt < max_retries - 1:
                continue

    logger.warning("LLM generation failed, degrading to rule-based")
    return None


@staticmethod
def _parse_llm_output(raw: str) -> tuple[str, str, float]:
    """解析 LLM 输出为 (observation, mental_model, confidence)。

    支持多种标记格式：
    1. 中文标记：【观察】/【心智模型】/【置信度】
    2. 英文标记：observation:/mental_model:/confidence:
    3. JSON 格式 {"observation": "...", "mental_model": "...", "confidence": 0.8}
    """
    observation = ""
    mental_model = ""
    confidence = None

    # 先尝试 JSON 格式解析
    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                observation = parsed.get("observation", parsed.get("obs", ""))
                mental_model = parsed.get("mental_model", parsed.get("model", ""))
                conf_val = parsed.get("confidence", parsed.get("conf", None))
                if conf_val is not None:
                    try:
                        confidence = max(0.0, min(1.0, float(conf_val)))
                    except (ValueError, TypeError):
                        confidence = 0.5
                if observation or mental_model:
                    if confidence is None:
                        confidence = 0.5
                    return (str(observation), str(mental_model), confidence)
        except (json.JSONDecodeError, ValueError):
            logger.debug("ReflectionEngine: JSON parse of LLM reflection output failed", exc_info=True)

    # 尝试中文标记解析
    obs_match = re.search(
        r"(?:【观察】|观察[：:]\s*)\s*\n?(.*?)(?=【心智模型】|心智模型[：:]|\Z)", raw, re.DOTALL
    )
    model_match = re.search(
        r"(?:【心智模型】|心智模型[：:]\s*)\s*\n?(.*?)(?=【置信度】|置信度[：:]|\Z)",
        raw,
        re.DOTALL,
    )
    conf_match = re.search(r"(?:【置信度】|置信度[：:]\s*)\s*\n?([\d.]+)", raw)

    if obs_match:
        observation = obs_match.group(1).strip()
    if model_match:
        mental_model = model_match.group(1).strip()
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5

    # 如果格式解析失败，整体作为观察
    if not observation and not mental_model:
        observation = raw.strip()[:500]

    # 未解析到置信度时使用默认值
    if confidence is None:
        confidence = 0.5

    return (observation, mental_model, confidence)
