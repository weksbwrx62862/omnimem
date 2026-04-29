#!/usr/bin/env python3
"""QUAL-2验证脚本：长文本子主题检索盲区修复验证。

测试场景：
- 写入包含量子计算的长文本（含"墨子号卫星量子密钥分发"子主题）
- 搜索"墨子号卫星量子密钥分发"
- 验证：应在Top-5结果中找到该条目
"""

import sys
import tempfile
from pathlib import Path

# 添加项目路径（支持从项目根目录直接运行）
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from omnimem.retrieval.engine import HybridRetriever


def test_quantum_computing_retrieval():
    """测试量子计算长文本的子主题检索能力。"""
    print("=" * 70)
    print("QUAL-2 验证：长文本子主题检索盲区修复")
    print("=" * 70)

    # 使用临时目录避免污染生产数据
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        retriever = HybridRetriever(data_dir=data_dir)

        # 构造包含多个子主题的量子计算长文本
        quantum_text = """
量子计算是利用量子力学原理（如叠加和纠缠）进行信息处理的革命性技术。
与传统计算机使用比特不同，量子计算机使用量子比特，可以同时处于0和1的叠加态。

## 量子算法

Shor算法能够快速分解大整数，对RSA加密构成威胁。Grover搜索算法可以在未排序数据库中实现平方级加速。

## 量子通信与密钥分发

量子密钥分发(QKD)利用量子力学原理实现理论上无条件安全的通信。
墨子号卫星是中国发射的世界首颗量子科学实验卫星，于2016年8月16日发射升空。
墨子号卫星实现了卫星与地面之间的量子密钥分发，通信距离超过1200公里，
创造了量子通信距离的世界纪录。该卫星由潘建伟院士团队主导研制。

## 量子硬件平台

超导量子计算：IBM、Google等公司采用超导电路实现量子比特。
离子阱量子计算：Honeywell/IonQ使用囚禁离子技术。
拓扑量子计算：微软致力于拓扑量子比特的研究。

## 应用前景

药物发现、材料设计、金融建模、人工智能优化等领域都将受益于量子计算的发展。
"""

        memory_id = "quantum-computing-001"
        metadata = {
            "source": "test",
            "topic": "量子计算",
            "created_at": "2024-01-01",
        }

        # 写入文档
        print("\n[步骤1] 写入量子计算长文本...")
        print(f"  文档长度: {len(quantum_text)} 字符")
        retriever.add(quantum_text, memory_id, metadata)
        print(f"  ✓ 文档已写入 (ID: {memory_id})")

        # 测试查询：子主题检索
        test_queries = [
            "墨子号卫星量子密钥分发",
            "潘建伟量子卫星",
            "Shor算法RSA加密",
            "超导量子比特",
        ]

        print("\n[步骤2] 执行子主题检索测试...")
        all_passed = True

        for query in test_queries:
            print(f"\n{'─' * 60}")
            print(f"查询: \"{query}\"")
            results = retriever.search(query, top_k=10, max_tokens=5000)
            print(f"返回结果数: {len(results)}")

            if not results:
                print("❌ 失败: 未返回任何结果")
                all_passed = False
                continue

            # 检查是否在Top-5中找到目标文档
            found = False
            found_rank = None
            for i, r in enumerate(results[:5], start=1):
                mid = r.get("memory_id", "")
                score = r.get("score", 0)
                content_preview = r.get("content", "")[:80].replace('\n', ' ')
                is_target = (mid == memory_id or mid.startswith(memory_id))
                marker = "★ 目标" if is_target else "  "
                print(f"  {marker} #{i} [score={score:.4f}] {mid}: {content_preview}...")
                if is_target and not found:
                    found = True
                    found_rank = i

            if found:
                print(f"✓ 成功: 目标文档排名 #{found_rank} (Top-5内)")
            else:
                print(f"❌ 失败: 目标文档未进入Top-5")
                # 检查是否在更后面的位置
                for i, r in enumerate(results[5:], start=6):
                    if r.get("memory_id", "") == memory_id or r.get("memory_id", "").startswith(memory_id):
                        print(f"  ⚠ 目标文档排名 #{i} (超出Top-5)")
                        break
                else:
                    print(f"  ✗ 目标文档完全未命中")
                all_passed = False

        # 输出总结
        print("\n" + "=" * 70)
        if all_passed:
            print("✅ QUAL-2 验证通过：所有子主题查询均在Top-5内命中目标文档")
            return 0
        else:
            print("❌ QUAL-2 验证失败：部分子主题查询未能正确检索")
            return 1


if __name__ == "__main__":
    sys.exit(test_quantum_computing_retrieval())
