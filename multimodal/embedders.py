"""媒体嵌入器与工厂（改进项 #12）。

- HashEmbedder：无模型的确定性嵌入（sha256 派生 → 定长 L2 归一向量），供离线/
  无凭证环境与测试使用；语义弱但可复现、满足 MediaEmbedder 协议。
- ClipEmbedder：真实图文跨模态嵌入，惰性导入 open_clip/torch，缺依赖时抛
  RuntimeError 并提示 `pip install 'omnimem[multimodal]'`（对齐 Neo4j 后端的惰性加载风格）。
- create_media_embedder：按 backend 名返回嵌入器（对齐 create_vector_store / create_graph_store）。
"""
from __future__ import annotations

import hashlib
import io
import logging
from typing import Any

from omnimem.multimodal.types import MediaEmbedder

logger = logging.getLogger(__name__)

SUPPORTED_MEDIA_BACKENDS = ("hash", "clip")


class HashEmbedder:
    """确定性哈希嵌入器（离线兜底，无外部依赖）。"""

    modality = "any"

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = max(1, int(dimension))

    def embed(self, blob: bytes) -> list[float]:
        seed = hashlib.sha256(blob or b"").digest()
        vals: list[float] = []
        counter = 0
        while len(vals) < self.dimension:
            block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            counter += 1
            for b in block:
                vals.append((b / 255.0) * 2.0 - 1.0)
                if len(vals) >= self.dimension:
                    break
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]


class ClipEmbedder:
    """CLIP 图文跨模态嵌入器（惰性导入依赖）。"""

    modality = "image"

    def __init__(
        self,
        model: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        try:
            import open_clip  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as e:  # ImportError 或子依赖缺失
            raise RuntimeError(
                "CLIP 媒体嵌入需要可选依赖，请安装：pip install 'omnimem[multimodal]'"
            ) from e

        self._torch = torch
        self._image_mod = Image
        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        self._model = self._model.to(device).eval()
        self._tokenizer = open_clip.get_tokenizer(model)
        self._dimension = self._infer_dimension()

    def _infer_dimension(self) -> int:
        dummy = self._image_mod.new("RGB", (224, 224))
        tensor = self._preprocess(dummy).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            feat = self._model.encode_image(tensor)
        return int(feat.shape[-1])

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, blob: bytes) -> list[float]:
        img = self._image_mod.open(io.BytesIO(blob)).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            feat = self._model.encode_image(tensor)[0]
        norm = feat.norm() or 1.0
        feat = feat / norm
        return [float(x) for x in feat.tolist()]


def create_media_embedder(backend: str = "hash", **kwargs: Any) -> MediaEmbedder:
    """构造媒体嵌入器。

    Args:
        backend: hash | clip
        kwargs: 透传给具体实现（dimension / model / pretrained / device）
    """
    if backend == "hash":
        return HashEmbedder(**kwargs)
    if backend == "clip":
        return ClipEmbedder(**kwargs)
    raise ValueError(
        f"Unknown media embedder backend: {backend} (可选: {', '.join(SUPPORTED_MEDIA_BACKENDS)})"
    )
