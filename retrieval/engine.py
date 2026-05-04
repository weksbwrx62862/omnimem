"""HybridRetriever — 混合检索编排。

6通道并行检索 + RRF 融合 + 可选 Cross-Encoder Rerank：
  1. 向量检索 (ChromaDB)
  2. BM25 关键词检索
  3. 目录检索 (Wing/Hall/Room 结构过滤)
  4. 实体提升 (Phase 3)
  5. 时间检索 (Phase 3)
  6. 图谱检索 (Phase 3)

Phase 1-2 实现: 向量 + BM25 + RRF 融合

读写锁优化：search() 用读锁（可并行），add() 用写锁（独占），
避免后台 queue_prefetch 写入阻塞主线程 prefetch 搜索。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from omnimem.retrieval.bm25 import BM25Retriever
from omnimem.retrieval.reranker import CrossEncoderReranker
from omnimem.retrieval.rrf import RRFFusion
from omnimem.retrieval.vector import VectorRetriever

logger = logging.getLogger(__name__)


class _ReadWriteLock:
    """简单的读写锁实现。

    多个读者可并行持有读锁；写者必须独占。
    写者优先策略：有写者等待时，新读者排队，防止写者饥饿。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._writer_waiting = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writers > 0 or self._writer_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writer_waiting += 1
            while self._readers > 0 or self._writers > 0:
                self._cond.wait()
            self._writer_waiting -= 1
            self._writers += 1

    def release_write(self) -> None:
        with self._cond:
            self._writers -= 1
            self._cond.notify_all()

    def __enter__(self) -> _ReadWriteLock:
        self.acquire_write()
        return self

    def __exit__(self, *args: object) -> None:
        self.release_write()


class HybridRetriever:
    """混合检索编排：向量 + BM25 + RRF 融合。

    6通道并行检索 + RRF 融合 + 可选 Cross-Encoder Rerank:
      1. 向量检索 (ChromaDB) — 语义相似度
      2. BM25 关键词检索 — 词袋匹配 + 同义词扩展
      3. 目录检索 (Wing/Hall/Room 结构过滤) — Phase 3
      4. 实体提升 — Phase 3
      5. 时间检索 — Phase 3
      6. 图谱检索 — Phase 3

    读写锁优化: search() 用读锁（可并行），add() 用写锁（独占），
    避免后台 queue_prefetch 写入阻塞主线程 prefetch 搜索。

    查询缓存: 相同查询 60s 内复用结果，写入时清除缓存。
    """

    _QUERY_CACHE_TTL = 60.0

    # ★ 类级别同义词映射：避免每次 _bm25_search 调用时重建大字典
    _SYNONYM_MAP = {
        # ─── 宠物领域（QUAL-3核心修复） ───
        "宠物": [
            "猫咪",
            "狗狗",
            "猫",
            "狗",
            "兔子",
            "仓鼠",
            "鹦鹉",
            "橘猫",
            "英短",
            "布偶",
            "缅因",
            "暹罗",
            "蓝猫",
            "加菲",
            "波斯猫",
            "美短",
            "折耳",
            "狸花",
            "三花",
            "奶牛猫",
            "金毛",
            "拉布拉多",
            "哈士奇",
            "泰迪",
            "柯基",
            "柴犬",
            "边牧",
            "萨摩耶",
            "阿拉斯加",
            "松狮",
            "比熊",
            "雪纳瑞",
        ],
        "猫咪": ["猫", "宠物", "喵星人", "主子", "橘猫", "英短", "布偶", "缅因", "暹罗"],
        "狗狗": ["狗", "宠物", "汪星人", "金毛", "拉布拉多", "哈士奇", "泰迪", "柯基"],
        "猫": ["猫咪", "宠物", "橘猫", "英短", "布偶", "缅因", "暹罗", "蓝猫"],
        "狗": ["狗狗", "宠物", "金毛", "拉布拉多", "哈士奇", "泰迪", "柯基", "柴犬"],
        # ─── 饮食领域 ───
        "饮食": [
            "食用",
            "喂食",
            "饲料",
            "吃",
            "食物",
            "营养",
            "猫粮",
            "狗粮",
            "罐头",
            "冻干",
            "猫条",
            "零食",
            "鸡胸肉",
            "牛肉",
            "鱼肉",
            "三文鱼",
            "虾",
            "生骨肉",
            "自制粮",
            "处方粮",
            "幼猫粮",
            "成猫粮",
            "化毛膏",
            "卵磷脂",
            "鱼油",
            "营养膏",
            "益生菌",
        ],
        "吃饭": ["饮食", "喂食", "吃", "食物", "猫粮", "狗粮", "罐头"],
        "喂食": ["饮食", "吃饭", "喂养", "投喂", "给吃的"],
        # ─── 技术领域（保留原有） ───
        "编程": ["代码", "开发", "程序", "coding", "写代码", "敲代码", "软件开发"],
        "部署": ["deploy", "上线", "发布", "运维", "发布上线", "生产环境"],
        "代码": ["编程", "开发", "程序", "coding", "源码", "脚本"],
        # ─── 个人信息领域 ───
        "姓名": ["名字", "称呼", "叫什么", "名号"],
        "城市": ["住址", "所在地", "地方", "位置", "居住"],
        "爱好": ["兴趣", "喜欢", "特长", "擅长", "业余"],
        "职业": ["工作", "行业", "岗位", "职位", "公司"],
        # ─── 通用高频词扩展 ───
        "不喜欢": ["讨厌", "反感", "拒绝", "不要", "别"],
        "喜欢": ["爱", "爱好", "感兴趣", "钟爱", "偏爱"],
        "问题": ["bug", "错误", "故障", "异常", "缺陷", "issue"],
        # ─── 深度学习领域（QUAL-2修复） ───
        "深度学习": [
            "神经网络",
            "深度神经",
            "CNN",
            "RNN",
            "Transformer",
            "alexnet",
            "resnet",
            "vggnet",
            "bert",
            "gpt",
            "机器学习",
            "训练",
            "推理",
            "模型",
        ],
        "神经网络": [
            "深度学习",
            "CNN",
            "RNN",
            "Transformer",
            "alexnet",
            "resnet",
            "感知机",
            "前馈",
            "循环",
        ],
        "机器学习": [
            "深度学习",
            "训练",
            "分类",
            "回归",
            "聚类",
            "特征",
            "模型",
            "监督学习",
            "无监督",
        ],
    }

    # ★ 类级别垃圾查询白名单：避免每次 _is_garbage_query 调用时重建集合
    _GARBAGE_COMMON_WORDS = frozenset(
        {
            "test",
            "what",
            "how",
            "why",
            "when",
            "where",
            "this",
            "that",
            "with",
            "from",
            "have",
            "will",
            "would",
            "could",
            "should",
            "about",
            "just",
            "like",
            "only",
            "some",
            "them",
            "than",
            "into",
            "over",
            "also",
            "back",
            "after",
            "used",
            "first",
            "well",
            "way",
            "even",
            "want",
            "because",
            "any",
            "these",
            "most",
            "make",
            "know",
            "time",
            "year",
            "good",
            "work",
            "qual",
            "data",
            "info",
            "user",
            "name",
            "code",
            "file",
            "http",
            "html",
            "json",
            "api",
            "url",
            "app",
            "log",
        }
    )

    def __init__(
        self,
        vector_backend: str = "chromadb",
        data_dir: Path | None = None,
        enable_reranker: bool = False,
    ):
        """初始化混合检索引擎。

        Args:
            vector_backend: 向量存储后端 (chromadb/qdrant/pgvector)
            data_dir: 检索数据存储目录
            enable_reranker: 是否启用 Cross-Encoder 重排序
        """
        self._data_dir = data_dir or Path("/tmp/omnimem/retrieval")
        self._vector = VectorRetriever(backend=vector_backend, data_dir=self._data_dir)
        self._bm25 = BM25Retriever(data_dir=self._data_dir)
        self._rrf = RRFFusion(k=60, min_rrf=0.035)
        self._reranker = CrossEncoderReranker() if enable_reranker else None
        # ★ 读写锁替代全局互斥锁
        self._rw_lock = _ReadWriteLock()
        # ★ 查询结果缓存：key → (results, timestamp)
        self._query_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
        # ★ P1方案四：动态来源权重（由 FeedbackCollector 驱动）
        self._source_weights: dict[str, float] = {}

    def embed_text(self, text: str) -> list[float]:
        """Embed text using the vector retriever."""
        return self._vector.embed_text(text)

    def add(self, content: str, memory_id: str, metadata: dict[str, Any]) -> None:
        """添加文档到所有检索通道。"""
        self._rw_lock.acquire_write()
        try:
            self._query_cache.clear()
            self._vector.add(content, memory_id, metadata)
            self._bm25.add(content, memory_id, metadata)
        finally:
            self._rw_lock.release_write()

    def add_batch(self, documents: list[dict[str, Any]]) -> None:
        """批量添加文档到所有检索通道。

        Args:
            documents: 文档列表，每项需包含 content 和 memory_id 字段
        """
        self._rw_lock.acquire_write()
        try:
            self._query_cache.clear()
            self._vector.add_batch(documents)
            self._bm25.add_batch(documents)
        finally:
            self._rw_lock.release_write()

    def search(
        self,
        query: str,
        max_tokens: int = 1500,
        mode: str = "rag",
        top_k: int = 10,
        store: Any = None,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """混合检索：向量 + BM25 + RRF 融合。

        RRF 融合检索流程:
          1. 查询缓存检查（60s TTL）
          2. 垃圾查询检测 → 限制 top_k 和 max_tokens
          3. 向量检索通道: ChromaDB 语义搜索
          4. BM25 检索通道: 关键词搜索 + 同义词扩展
          5. RRF 融合: 合并两路结果，数据量自适应 min_rrf 阈值
          6. 垃圾查询二次验证: 低分结果过滤
          7. 可选 Cross-Encoder Rerank
          8. Token 预算裁剪

        mode:
          rag: 快速向量+BM25混合检索（毫秒级）
          llm: 深度检索 — 更多结果 + store 内容搜索补充通道

        Args:
            query: 检索查询文本
            max_tokens: 返回结果的最大 token 预算
            mode: 检索模式 (rag/llm)
            top_k: 每个通道返回的最大结果数
            store: DrawerClosetStore 实例（保留参数，暂未使用）

        Returns:
            检索结果列表，每项包含 content/memory_id/score/metadata 等字段
        """
        is_garbage = self._is_garbage_query(query)

        try:
            doc_count = self._vector.count()
        except Exception:
            doc_count = 0

        if is_garbage:
            top_k = min(top_k, 2)
            max_tokens = min(max_tokens, 200)
            if doc_count >= 100:
                top_k = 1
            elif doc_count >= 30:
                top_k = min(top_k, 1)

        if mode == "llm" and not is_garbage:
            top_k = max(top_k, 20)
            max_tokens = max(max_tokens, 3000)

        self._rw_lock.acquire_read()
        try:
            # ★ 查询缓存检查（在读锁内，避免与 add 清缓存竞态）
            cache_key = f"{query}|{max_tokens}|{mode}|{top_k}"
            now = time.time()
            if cache_key in self._query_cache:
                cached_results, cached_time = self._query_cache[cache_key]
                if now - cached_time < self._QUERY_CACHE_TTL:
                    logger.debug("HybridRetriever query cache hit: %s", query[:50])
                    return cached_results

            # ★ 并行执行向量检索与 BM25 检索，降低搜索延迟
            with ThreadPoolExecutor(max_workers=2) as executor:
                vec_future = executor.submit(self._vector_search, query, top_k)
                bm25_future = executor.submit(self._bm25_search, query, top_k)
                vector_results = vec_future.result()
                bm25_results = bm25_future.result()
            results = self._rrf_fuse(
                query,
                vector_results,
                bm25_results,
                is_garbage=is_garbage,
                doc_count=doc_count,
                top_k=top_k,
                max_tokens=max_tokens,
            )
            # ★ 缓存搜索结果
            self._query_cache[cache_key] = (results, now)
            return results
        finally:
            self._rw_lock.release_read()

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """向量检索通道。"""
        return self._vector.search(query, top_k=top_k)

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25 检索通道（含同义词扩展）。

        ★ 同义词扩展 BM25 查询：弥补词袋模型的语义鸿沟（QUAL-3修复）
        注意：单字会被 _tokenize 丢弃，所以用2+字词
        ★ 扩展策略：上位词↔下位词 双向扩展 + 品种级细粒度覆盖
        """
        bm25_results = self._bm25.search(query, top_k=top_k)
        for key, synonyms in self._SYNONYM_MAP.items():
            if key in query:
                for syn in synonyms:
                    expanded = query.replace(key, syn)
                    expanded_results = self._bm25.search(expanded, top_k=top_k)
                    existing_ids = {r.get("memory_id", "") for r in bm25_results}
                    for r in expanded_results:
                        if r.get("memory_id", "") not in existing_ids:
                            bm25_results.append(r)
                            existing_ids.add(r.get("memory_id", ""))
        return bm25_results

    def _rrf_fuse(
        self,
        query: str,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        *,
        is_garbage: bool,
        doc_count: int,
        top_k: int,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """RRF 融合 + 数据量自适应阈值 + 垃圾查询二次验证 + Rerank + Token 裁剪。"""
        # 数据量自适应 min_rrf
        # ★ R22修复：RRF 分数数学上限 = weight/(k+rank)
        # 向量 rank1: 3.0/61=0.0492, +bonus(x1.5)=0.0738
        # BM25 rank1: 1.0/61=0.0164
        # 双路最优: 0.0738+0.0164=0.0902
        # 阈值不能超过数学上限，否则所有结果被过滤
        adaptive_min_rrf = 0.035
        if doc_count >= 200 or doc_count >= 100:
            adaptive_min_rrf = 0.04
        elif doc_count >= 50 or doc_count >= 20:
            adaptive_min_rrf = 0.035
        # ★ R25优化：语料过少时（<10条）降低阈值
        # 小语料下 BM25 IDF 不可靠，单通道匹配的 RRF 分数低
        # 如仅 BM25 rank1: 1.0/61=0.0164 < 0.035 会被误过滤
        elif doc_count < 10:
            adaptive_min_rrf = 0.01
        # ★ 单通道降权：当某通道返回空结果时，降低 min_rrf 阈值
        # 避免单通道结果因 RRF 分数不足被全部过滤
        # ★ R26修复：之前的条件 adaptive_min_rrf > 0.035 永远为 False
        # （只有 doc_count >= 50 时 adaptive_min_rrf 才会 >= 0.04，但此时不应降权）
        # 修正：当仅单通道有结果时，在非大语料场景下降权
        single_channel = (not vector_results) or (not bm25_results)
        if single_channel and doc_count < 50 and adaptive_min_rrf > 0.01:
            adaptive_min_rrf = 0.01
        # ★ P1方案四：应用动态来源权重（基于 FeedbackCollector 的 CTR 统计）
        base_weights = [3.0, 1.0]
        if self._source_weights:
            base_weights[0] *= self._source_weights.get("vector", 1.0)
            base_weights[1] *= self._source_weights.get("bm25", 1.0)
        fused = self._rrf.merge(
            [vector_results, bm25_results],
            min_rrf=adaptive_min_rrf,
            weights=base_weights,
        )

        # ★ QUAL-1 R14/R24修复：垃圾查询结果二次验证
        # 垃圾查询（如 zzzzzxyz123）不应返回任何结果
        if is_garbage and fused:
            # ★ R24修复：垃圾查询直接清空结果
            # 之前的阈值 0.02 过于宽松，向量搜索总返回 top_k 个最近邻，
            # 即使查询无意义，score 仍可能达到 0.05+
            fused = []

        # 可选 Rerank
        if self._reranker and len(fused) > 3:
            fused = self._reranker.rerank(query, fused, top_k=top_k)

        # Token 预算裁剪
        return self._trim_to_budget(fused, max_tokens)

    def index_update(self, user_content: str, assistant_content: str) -> None:  # noqa: ARG002
        """后台异步索引更新（从 sync_turn 调用）。

        ★ 只存提炼后的用户消息核心事实，不存 "User:...Assistant:..." 对话原文。
        对话原文被存入 BM25 会导致 recall 返回大段对话片段而非事实。
        """
        import re
        import uuid

        clean_user = re.sub(
            r"### Relevant Memories(?:\s*\(prefetched\))?\s*\n.*"
            r"(?=\n(?!- )|\Z)",
            "",
            user_content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        clean_user = re.sub(r"^- \[cached\].*$", "", clean_user, flags=re.MULTILINE).strip()

        # ★ 只存用户消息的提炼摘要，不存对话原文
        # 截取用户消息的核心部分（前200字），不拼 Assistant 回复
        content = clean_user[:200].strip() if clean_user else ""
        if not content or len(content) < 5:
            return  # 空或极短内容不索引

        idx_id = f"sync-{uuid.uuid4().hex[:8]}"
        self._rw_lock.acquire_write()
        try:
            self._query_cache.clear()
            self._vector.add(content, memory_id=idx_id, metadata={"source": "sync_turn"})
            self._bm25.add(content, memory_id=idx_id, metadata={"source": "sync_turn"})
        finally:
            self._rw_lock.release_write()

    def flush(self) -> None:
        """刷新所有索引。"""
        self._vector.flush()
        self._bm25.flush()

    def set_source_weights(self, weights: dict[str, float]) -> None:
        """设置动态来源权重（由 FeedbackCollector 驱动）。

        Args:
            weights: 来源类型 → 权重映射，如 {"vector": 3.5, "bm25": 0.8}
        """
        self._source_weights = dict(weights)

    def invalidate_cache(self) -> None:
        """清除查询结果缓存（写入时调用）。"""
        self._query_cache.clear()

    @property
    def bm25_document_count(self) -> int:
        """BM25 已索引文档数（用于判断是否需要跨会话重建）。"""
        return self._bm25.document_count

    def rebuild_bm25_from_entries(self, entries: list[dict[str, Any]]) -> int:
        """从索引条目重建 BM25 检索通道（跨会话持久化恢复）。

        若 BM25 已从磁盘缓存恢复且有数据，跳过重复重建。
        否则委托给 BM25Retriever.rebuild_from_entries 进行全量重建。

        Args:
            entries: 索引条目列表，需含 content/summary, memory_id, type, scope 等字段

        Returns:
            重建的条目数
        """
        if self._bm25.cache_loaded and self._bm25.document_count > 0:
            logger.debug(
                "BM25 already has %d entries from disk cache, skipping rebuild",
                self._bm25.document_count,
            )
            return 0
        return self._bm25.rebuild_from_entries(entries)

    @staticmethod
    def _is_garbage_query(query: str) -> bool:
        """检测查询是否为无意义/垃圾输入（QUAL-1修复）。

        以下情况判定为垃圾查询：
        1. 纯随机字符串（连续5+非词典字符且无中文/常见英文单词）
        2. 极短查询（<2字符）且无中文
        3. 纯数字/纯符号串
        4. 包含少量常见词但主体为随机字符（如 zzzzzxyz123test）

        Returns:
            True 表示应限制返回结果数量
        """
        import re

        q = query.strip()
        if not q or len(q) < 2:
            return True

        # 有中文字符 → 不是垃圾
        if re.search(r"[\u4e00-\u9fff]", q):
            return False

        words = re.findall(r"[a-zA-Z]{3,}", q.lower())
        word_set = set(words)
        matched_common = word_set & HybridRetriever._GARBAGE_COMMON_WORDS

        # 规则A: 多个(≥2)常见词构成有意义句子 → 非垃圾
        # 但要求常见词覆盖的总字符占比>40%，防止随机串中嵌入少量词典词
        if len(matched_common) >= 2:
            common_char_len = sum(len(w) for w in words if w in matched_common)
            if common_char_len / len(q) > 0.4 and len(word_set) <= len(matched_common) + 2:
                return False

        # 规则B: 单个常见词但周围全是随机字符 → 垃圾
        # 如 "zzzzzxyz123qual1test": 只有test/qual匹配，其余是噪声
        if matched_common and len(q) > 8:
            non_word_chars = re.sub(r"[a-zA-Z]{3,}", "", q)
            noise_ratio = len(non_word_chars) / len(q)
            if noise_ratio > 0.5:
                return True

        # 规则C: 连续随机字符比例 > 60%
        random_chars = re.sub(r"[a-zA-Z0-9\s]", "", q)
        if len(random_chars) > len(q) * 0.6:
            return True

        # 规则D: 纯数字串
        if re.match(r"^[\d\s]+$", q):
            return True

        # 规则E: 连续5+无元音或高重复字母序列
        alpha_seq = re.findall(r"[a-zA-Z]{5,}", q)
        for seq in alpha_seq:
            seq_lower = seq.lower()
            if seq_lower not in HybridRetriever._GARBAGE_COMMON_WORDS:
                vowel_count = sum(1 for c in seq_lower if c in "aeiou")
                unique_chars = len(set(seq_lower))
                if vowel_count == 0 or unique_chars <= 2:
                    return True

        # 规则F: 仅1个短常见词且总长<5 → 垃圾（如 "test", "info"）
        if len(matched_common) == 1 and len(q) < 6 and len(word_set) <= 1:
            return True

        # 规则G: 纯字母数字长串(>8)且无任何常见词匹配 → 垃圾
        # 如 "abcdefg123456", "asdfghjkl123"
        return not matched_common and re.match(r"^[a-zA-Z0-9]+$", q) is not None and len(q) > 8

    @staticmethod
    def _trim_to_budget(results: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
        """裁剪结果到 Token 预算内。"""
        budget = max_tokens
        chars_per_token = 4
        trimmed = []
        used = 0
        for r in results:
            content = r.get("content", "")
            est_tokens = max(1, len(content) // chars_per_token)
            if used + est_tokens <= budget:
                trimmed.append(r)
                used += est_tokens
            else:
                # ★ 超预算但不是break — 跳过超大条目继续看后面的
                # 之前的break会导致排在前面的超大条目吃掉整个预算
                continue
        return trimmed
