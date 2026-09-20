"""core/block.py CoreBlock 单测（离线）。"""
from __future__ import annotations

from omnimem.core.block import CoreBlock


def test_empty_block_renders_blank():
    assert CoreBlock().to_prompt_text() == ""


def test_update_and_render_sections():
    b = CoreBlock(identity_block="我是助手")
    b.update_context("当前在做导入")
    b.update_plan("1. 转换 2. 落库")
    text = b.to_prompt_text()
    assert "### Identity\n我是助手" in text
    assert "### Current Context\n当前在做导入" in text
    assert "### Plan\n1. 转换 2. 落库" in text
    assert text.count("\n\n") == 2  # 三段之间两个分隔


def test_partial_sections_only_present():
    b = CoreBlock(context_block="ctx")
    text = b.to_prompt_text()
    assert text == "### Current Context\nctx"
    assert "Identity" not in text and "Plan" not in text


def test_update_mutates_fields():
    b = CoreBlock()
    b.update_plan("p1")
    assert b.plan_block == "p1"
    b.update_context("c1")
    assert b.context_block == "c1"
