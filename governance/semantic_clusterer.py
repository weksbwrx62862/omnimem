"""
SemanticClusterer — 语义聚类模块。

基于向量的记忆聚类分析：
1. K-Means 聚类
2. DBSCAN 聚类
3. 聚类中心识别
4. 异常记忆检测

核心算法:
- K-Means: 划分 K 个簇，最小化簇内距离
- DBSCAN: 基于密度的聚类，自动发现异常值
- 聚类中心: 簇内所有向量的均值
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """聚类簇"""
    cluster_id: int
    center: list[float]
    members: list[str]
    size: int
    avg_distance: float


@dataclass
class ClusteringResult:
    """聚类结果"""
    clusters: list[Cluster]
    outliers: list[str]
    silhouette_score: float
    total_memories: int


class SemanticClusterer:
    """语义聚类器

    提供:
    - K-Means 聚类
    - DBSCAN 聚类
    - 聚类中心识别
    - 异常记忆检测
    """

    def __init__(
        self,
        embedding_path: Optional[str] = None,
    ):
        self._embedding_path = embedding_path or os.path.expanduser(
            "~/.hermes/omnimem/retrieval/embedding_cache.json"
        )
        self._embeddings: dict[str, list[float]] = {}
        self._loaded = False

    def _load_embeddings(self) -> None:
        """加载嵌入向量"""
        if self._loaded:
            return

        try:
            if os.path.exists(self._embedding_path):
                with open(self._embedding_path) as f:
                    self._embeddings = json.load(f)
                logger.info("Loaded %d embeddings", len(self._embeddings))
        except Exception as e:
            logger.warning("Failed to load embeddings: %s", e)

        self._loaded = True

    def kmeans(
        self,
        k: int = 5,
        max_iterations: int = 100,
        tolerance: float = 0.001,
    ) -> ClusteringResult:
        """K-Means 聚类

        Args:
            k: 簇数量
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值

        Returns:
            聚类结果
        """
        self._load_embeddings()

        if len(self._embeddings) < k:
            return ClusteringResult(clusters=[], outliers=[], silhouette_score=0.0, total_memories=0)

        # 准备数据
        memory_ids = list(self._embeddings.keys())
        vectors = [self._embeddings[mid] for mid in memory_ids]
        dim = len(vectors[0]) if vectors else 0

        # 随机初始化聚类中心
        centers = random.sample(vectors, k)

        # 迭代优化
        for iteration in range(max_iterations):
            # 分配每个点到最近的聚类中心
            clusters: list[list[int]] = [[] for _ in range(k)]
            for i, vec in enumerate(vectors):
                distances = [self._euclidean_distance(vec, center) for center in centers]
                cluster_idx = distances.index(min(distances))
                clusters[cluster_idx].append(i)

            # 更新聚类中心
            new_centers = []
            for cluster in clusters:
                if cluster:
                    # 计算均值
                    center = [0.0] * dim
                    for idx in cluster:
                        for d in range(dim):
                            center[d] += vectors[idx][d]
                    center = [c / len(cluster) for c in center]
                    new_centers.append(center)
                else:
                    # 空簇，随机重新初始化
                    new_centers.append(random.choice(vectors))

            # 检查收敛
            max_shift = 0
            for old, new in zip(centers, new_centers):
                shift = self._euclidean_distance(old, new)
                max_shift = max(max_shift, shift)

            centers = new_centers

            if max_shift < tolerance:
                logger.info("K-Means converged at iteration %d", iteration)
                break

        # 构建结果
        result_clusters = []
        for i, cluster_indices in enumerate(clusters):
            if cluster_indices:
                members = [memory_ids[idx] for idx in cluster_indices]
                center = centers[i]

                # 计算平均距离
                distances = [
                    self._euclidean_distance(vectors[idx], center)
                    for idx in cluster_indices
                ]
                avg_dist = sum(distances) / len(distances)

                result_clusters.append(Cluster(
                    cluster_id=i,
                    center=center,
                    members=members,
                    size=len(members),
                    avg_distance=avg_dist,
                ))

        # 计算轮廓系数
        silhouette = self._calculate_silhouette(vectors, clusters, k)

        return ClusteringResult(
            clusters=result_clusters,
            outliers=[],
            silhouette_score=silhouette,
            total_memories=len(memory_ids),
        )

    def dbscan(
        self,
        eps: float = 0.5,
        min_samples: int = 3,
    ) -> ClusteringResult:
        """DBSCAN 聚类

        Args:
            eps: 邻域半径
            min_samples: 最小样本数

        Returns:
            聚类结果
        """
        self._load_embeddings()

        if not self._embeddings:
            return ClusteringResult(clusters=[], outliers=[], silhouette_score=0.0, total_memories=0)

        # 准备数据
        memory_ids = list(self._embeddings.keys())
        vectors = [self._embeddings[mid] for mid in memory_ids]
        n = len(vectors)

        # 初始化标签 (-1 = 未访问, 0 = 噪声, >0 = 簇ID)
        labels = [-1] * n
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue

            # 找到邻域内的点
            neighbors = []
            for j in range(n):
                if self._euclidean_distance(vectors[i], vectors[j]) <= eps:
                    neighbors.append(j)

            if len(neighbors) < min_samples:
                labels[i] = 0  # 噪声
            else:
                # 创建新簇
                cluster_id += 1
                labels[i] = cluster_id

                # 扩展簇
                seed_set = list(neighbors)
                seed_set.remove(i)

                while seed_set:
                    j = seed_set.pop(0)

                    if labels[j] == 0:
                        labels[j] = cluster_id
                    elif labels[j] == -1:
                        labels[j] = cluster_id

                        # 找到 j 的邻域
                        j_neighbors = []
                        for k in range(n):
                            if self._euclidean_distance(vectors[j], vectors[k]) <= eps:
                                j_neighbors.append(k)

                        if len(j_neighbors) >= min_samples:
                            seed_set.extend(j_neighbors)

        # 构建结果
        clusters_dict: dict[int, list[int]] = {}
        outliers = []

        for i, label in enumerate(labels):
            if label == 0:
                outliers.append(memory_ids[i])
            elif label > 0:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(i)

        result_clusters = []
        for cluster_id, indices in clusters_dict.items():
            members = [memory_ids[idx] for idx in indices]

            # 计算聚类中心
            dim = len(vectors[0]) if vectors else 0
            center = [0.0] * dim
            for idx in indices:
                for d in range(dim):
                    center[d] += vectors[idx][d]
            center = [c / len(indices) for c in center]

            # 计算平均距离
            distances = [
                self._euclidean_distance(vectors[idx], center)
                for idx in indices
            ]
            avg_dist = sum(distances) / len(distances)

            result_clusters.append(Cluster(
                cluster_id=cluster_id,
                center=center,
                members=members,
                size=len(members),
                avg_distance=avg_dist,
            ))

        return ClusteringResult(
            clusters=result_clusters,
            outliers=outliers,
            silhouette_score=0.0,  # DBSCAN 不计算轮廓系数
            total_memories=n,
        )

    def _euclidean_distance(self, vec1: list[float], vec2: list[float]) -> float:
        """计算欧氏距离"""
        if len(vec1) != len(vec2):
            return float('inf')

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    def _calculate_silhouette(
        self,
        vectors: list[list[float]],
        clusters: list[list[int]],
        k: int,
    ) -> float:
        """计算轮廓系数

        轮廓系数范围: [-1, 1]
        - 接近 1: 聚类效果好
        - 接近 0: 聚类效果一般
        - 接近 -1: 聚类效果差
        """
        n = len(vectors)
        if n == 0 or k == 0:
            return 0.0

        # 为每个点分配簇标签
        labels = [0] * n
        for cluster_idx, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cluster_idx

        # 计算每个点的轮廓系数
        silhouette_values = []

        for i in range(n):
            # a(i): 点 i 到同簇其他点的平均距离
            same_cluster = [j for j in clusters[labels[i]] if j != i]
            if same_cluster:
                a_i = sum(self._euclidean_distance(vectors[i], vectors[j]) for j in same_cluster) / len(same_cluster)
            else:
                a_i = 0

            # b(i): 点 i 到最近其他簇的平均距离
            b_i = float('inf')
            for cluster_idx in range(k):
                if cluster_idx != labels[i]:
                    other_cluster = clusters[cluster_idx]
                    if other_cluster:
                        avg_dist = sum(self._euclidean_distance(vectors[i], vectors[j]) for j in other_cluster) / len(other_cluster)
                        b_i = min(b_i, avg_dist)

            if b_i == float('inf'):
                b_i = 0

            # 轮廓系数
            if max(a_i, b_i) == 0:
                silhouette_values.append(0)
            else:
                silhouette_values.append((b_i - a_i) / max(a_i, b_i))

        return sum(silhouette_values) / len(silhouette_values) if silhouette_values else 0.0

    def get_cluster_summary(self, result: ClusteringResult) -> dict[str, Any]:
        """获取聚类摘要

        Args:
            result: 聚类结果

        Returns:
            摘要信息
        """
        return {
            "total_memories": result.total_memories,
            "num_clusters": len(result.clusters),
            "num_outliers": len(result.outliers),
            "silhouette_score": result.silhouette_score,
            "clusters": [
                {
                    "id": c.cluster_id,
                    "size": c.size,
                    "avg_distance": c.avg_distance,
                    "members": c.members[:5],  # 只显示前5个
                }
                for c in result.clusters
            ],
            "outliers": result.outliers[:10],  # 只显示前10个
        }


# 全局实例
_clusterer: Optional[SemanticClusterer] = None


def get_clusterer(embedding_path: Optional[str] = None) -> SemanticClusterer:
    """获取全局聚类器实例"""
    global _clusterer
    if _clusterer is None:
        _clusterer = SemanticClusterer(embedding_path)
    return _clusterer


def cluster_memories(k: int = 5, method: str = "kmeans") -> dict[str, Any]:
    """便捷函数：聚类记忆"""
    clusterer = get_clusterer()

    if method == "kmeans":
        result = clusterer.kmeans(k=k)
    elif method == "dbscan":
        result = clusterer.dbscan()
    else:
        return {"error": f"Unknown method: {method}"}

    return clusterer.get_cluster_summary(result)
