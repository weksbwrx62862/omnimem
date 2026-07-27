#!/bin/bash
# OmniMem → LongMemEval-S 评测一键运行脚本
set -e

OMNIMEM_DIR="/home/xxh/.hermes/plugins/omnimem"
LME_DIR="$OMNIMEM_DIR/benchmarks/LongMemEval"
DATA_DIR="$LME_DIR/data"
RESULTS_DIR="$OMNIMEM_DIR/benchmarks/results/longmemeval"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  OmniMem LongMemEval-S 评测"
echo "============================================"
echo ""

# ── DashScope LLM API 配置 ──
# 优先使用环境变量，如果未设置则从 hermes config.yaml 读取
if [ -z "$OPENAI_API_KEY" ]; then
    # 尝试从 hermes config.yaml 读取 dashscope.api_key
    HERMES_CONFIG="$HOME/.hermes/config.yaml"
    if [ -f "$HERMES_CONFIG" ]; then
        DS_KEY=$(python3 -c "
import yaml
with open('$HERMES_CONFIG') as f:
    cfg = yaml.safe_load(f)
ds = cfg.get('providers', {}).get('dashscope', {})
print(ds.get('api_key', ''))
" 2>/dev/null || echo "")
        if [ -n "$DS_KEY" ]; then
            export OPENAI_API_KEY="$DS_KEY"
            export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
            echo "从 hermes config.yaml 读取 DashScope 凭证"
        fi
    fi
fi

# 检查数据文件
if [ ! -f "$DATA_DIR/longmemeval_s_cleaned.json" ]; then
    echo "错误: 数据文件不存在 $DATA_DIR/longmemeval_s_cleaned.json"
    exit 1
fi

# 检查 API 可用性
LLM_MODE="retrieval-only"
if [ -n "$OPENAI_API_KEY" ]; then
    LLM_MODE="full"
    echo "检测到 OPENAI_API_KEY，将运行完整评测（检索+生成+评判）"
else
    echo "未检测到 OPENAI_API_KEY，将运行 retrieval-only 模式（仅检索评估）"
fi
echo ""

# 解析参数
LIMIT="${1:-50}"
echo "评测参数:"
echo "  数据: longmemeval_s_cleaned.json"
echo "  限制题目数: $LIMIT"
echo "  模式: $LLM_MODE"
echo "  结果目录: $RESULTS_DIR"
echo ""

echo "步骤 1/4: Ingest（会话历史写入 OmniMem）"
echo "步骤 2/4: Retrieval（OmniMem 检索）"
if [ "$LLM_MODE" = "full" ]; then
    echo "步骤 3/4: Generation（LLM 生成答案）"
    echo "步骤 4/4: Evaluation（评判正确性）"
else
    echo "步骤 3/4: 跳过（retrieval-only 模式）"
    echo "步骤 4/4: 检索质量评估"
fi
echo ""

echo "--- 开始评测 ---"
cd "$OMNIMEM_DIR"

python3 benchmarks/run_longmemeval.py \
    --data "$DATA_DIR/longmemeval_s_cleaned.json" \
    --output "$RESULTS_DIR" \
    --limit "$LIMIT" \
    --top-k 10 \
    --gen-model qwen-plus \
    --judge-model qwen-plus

echo ""
echo "--- 评测完成 ---"
echo "结果文件: $RESULTS_DIR/scores.json"
echo "详细日志: $RESULTS_DIR/details.jsonl"
