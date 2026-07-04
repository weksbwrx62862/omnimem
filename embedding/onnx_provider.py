"""ONNX EmbeddingProvider 实现（可选依赖）。

onnxruntime 未安装时，构造阶段即抛出明确错误，便于工厂层优雅降级。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from omnimem.embedding.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ONNXEmbeddingProvider(EmbeddingProvider):
    """基于 ONNX Runtime 的本地嵌入服务。

    设计要点：
      - 延迟加载模型，避免未使用时的导入开销
      - 同步 embed 在线程池中执行前向推理，避免阻塞事件循环
      - 未安装 onnxruntime 时给出明确错误信息
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_path: str = "",
        dimensions: int | None = None,
    ):
        self._model_name = model_name
        self._model_path = model_path
        self._dimensions = dimensions
        self._session: Any = None
        self._tokenizer: Any = None

    @property
    def model_name(self) -> str:
        return self._model_path or self._model_name

    @property
    def dimension(self) -> int:
        """返回模型维度，优先使用用户指定值。"""
        if self._dimensions is not None:
            return self._dimensions
        try:
            session = self._get_session()
            return int(session.get_outputs()[0].shape[-1])
        except Exception as e:
            logger.warning("Failed to get ONNX embedding dimension: %s", e)
            return 0

    def _ensure_runtime(self) -> Any:
        """确保 onnxruntime 已安装，返回模块。"""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime 未安装，无法使用 ONNXEmbeddingProvider。"
                "请执行: pip install onnxruntime"
            ) from e
        return ort

    def _get_session(self) -> Any:
        """延迟初始化 ONNX InferenceSession。"""
        if self._session is None:
            ort = self._ensure_runtime()
            model_path = self._model_path or self._model_name
            logger.info("Loading ONNX model: %s", model_path)
            # 优先使用 GPU，不可用则回退 CPU
            providers = ort.get_available_providers()
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                model_path, sess_options=sess_options, providers=providers
            )
        return self._session

    def _tokenize(self, texts: list[str]) -> dict[str, Any]:
        """使用 transformers tokenizer 对文本编码。"""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise RuntimeError(
                    "使用 ONNXEmbeddingProvider 需要 transformers 库进行 tokenize。"
                    "请执行: pip install transformers"
                ) from e
            tokenizer_name = self._model_name
            # 若 model_path 指向 ONNX 文件，则尝试用模型名加载 tokenizer
            if self._model_path and self._model_path.endswith(".onnx"):
                tokenizer_name = self._model_name
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=512,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """同步推理获取 embeddings。"""
        if not texts:
            return []
        session = self._get_session()
        encoded = self._tokenize(texts)
        input_names = {inp.name for inp in session.get_inputs()}
        feed = {
            name: encoded[name]
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in input_names
        }
        outputs = session.run(None, feed)
        embeddings = outputs[0]
        # 对 [batch, seq, dim] 输出做 mean pooling（若已是 [batch, dim] 则直接返回）
        if len(embeddings.shape) == 3:
            attention_mask = encoded["attention_mask"]
            mask_expanded = attention_mask.reshape(
                attention_mask.shape[0], attention_mask.shape[1], 1
            )
            sum_embeddings = (embeddings * mask_expanded).sum(axis=1)
            embeddings = sum_embeddings / attention_mask.sum(axis=1, keepdims=True)
        return [vec.tolist() for vec in embeddings]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """异步包装：在线程池中执行同步推理。"""
        return await asyncio.to_thread(self.embed, texts)
