"""Tests for governance.semantic_clusterer — K-Means / DBSCAN / silhouette."""

from __future__ import annotations

import math

import pytest

from governance import semantic_clusterer as sc
from governance.semantic_clusterer import (
    Cluster,
    ClusteringResult,
    SemanticClusterer,
    cluster_memories,
    get_clusterer,
)


def _inject(clusterer: SemanticClusterer, embeddings: dict[str, list[float]]) -> SemanticClusterer:
    """Bypass _load_embeddings and inject vectors directly."""
    clusterer._embeddings = dict(embeddings)
    clusterer._loaded = True
    return clusterer


# ── dataclasses ──────────────────────────────────────────────────────────


def test_cluster_construction() -> None:
    c = Cluster(cluster_id=1, center=[0.5, 0.5], members=["a", "b"], size=2, avg_distance=0.1)
    assert c.cluster_id == 1
    assert c.center == [0.5, 0.5]
    assert c.members == ["a", "b"]
    assert c.size == 2
    assert c.avg_distance == 0.1


def test_clustering_result_construction() -> None:
    r = ClusteringResult(clusters=[], outliers=["x"], silhouette_score=0.5, total_memories=3)
    assert r.clusters == []
    assert r.outliers == ["x"]
    assert r.silhouette_score == 0.5
    assert r.total_memories == 3


# ── constructor ──────────────────────────────────────────────────────────


def test_default_embedding_path() -> None:
    s = SemanticClusterer()
    assert "embedding_cache.json" in s._embedding_path
    assert s._loaded is False
    assert s._embeddings == {}


def test_explicit_embedding_path(tmp_path) -> None:
    p = tmp_path / "emb.json"
    s = SemanticClusterer(embedding_path=str(p))
    assert s._embedding_path == str(p)


# ── _euclidean_distance ─────────────────────────────────────────────────


def test_euclidean_distance_zero(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    assert s._euclidean_distance([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_euclidean_distance_unit(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    assert s._euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_euclidean_distance_mismatched_dims_inf(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    assert s._euclidean_distance([1.0, 2.0], [1.0]) == float("inf")


def test_euclidean_distance_negative(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    d = s._euclidean_distance([-1.0, -1.0], [1.0, 1.0])
    assert d == pytest.approx(math.sqrt(8.0))


# ── _load_embeddings ────────────────────────────────────────────────────


def test_load_embeddings_missing_file_non_fatal(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "missing.json"))
    s._load_embeddings()
    assert s._loaded is True
    assert s._embeddings == {}


def test_load_embeddings_idempotent(tmp_path, monkeypatch) -> None:
    import omnimem.retrieval.vector_store as vs

    calls: list[str] = []

    def _fake_load(path):
        calls.append(str(path))
        return {"a": [1.0, 2.0]}

    monkeypatch.setattr(vs, "load_embedding_cache_dict", _fake_load)
    s = SemanticClusterer(embedding_path=str(tmp_path / "e.json"))
    s._load_embeddings()
    s._load_embeddings()
    assert len(calls) == 1


def test_load_embeddings_exception_swallowed(tmp_path, monkeypatch) -> None:
    import omnimem.retrieval.vector_store as vs

    def _boom(path):
        raise RuntimeError("nope")

    monkeypatch.setattr(vs, "load_embedding_cache_dict", _boom)
    s = SemanticClusterer(embedding_path=str(tmp_path / "e.json"))
    s._load_embeddings()  # should not raise
    assert s._loaded is True
    assert s._embeddings == {}


# ── kmeans ──────────────────────────────────────────────────────────────


def test_kmeans_too_few_embeddings(tmp_path) -> None:
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), {"a": [1.0, 0.0]})
    r = s.kmeans(k=5)
    assert r.clusters == []
    assert r.total_memories == 0


def test_kmeans_empty_embeddings(tmp_path) -> None:
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), {})
    r = s.kmeans(k=3)
    assert r.clusters == []
    assert r.outliers == []
    assert r.total_memories == 0


def test_kmeans_two_clusters(tmp_path) -> None:
    data = {
        "a1": [0.0, 0.0],
        "a2": [0.1, 0.0],
        "a3": [0.0, 0.1],
        "b1": [10.0, 10.0],
        "b2": [10.1, 10.0],
        "b3": [10.0, 10.1],
    }
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.kmeans(k=2, max_iterations=50)
    assert r.total_memories == 6
    assert len(r.clusters) == 2
    sizes = sorted(c.size for c in r.clusters)
    assert sizes == [3, 3]


def test_kmeans_single_cluster(tmp_path) -> None:
    data = {"x": [1.0], "y": [1.1], "z": [0.9]}
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.kmeans(k=1, max_iterations=10)
    assert r.total_memories == 3
    assert len(r.clusters) == 1
    assert r.clusters[0].size == 3


def test_kmeans_silhouette_in_range(tmp_path) -> None:
    data = {
        "a1": [0.0, 0.0],
        "a2": [0.1, 0.0],
        "a3": [0.0, 0.1],
        "b1": [10.0, 10.0],
        "b2": [10.1, 10.0],
        "b3": [10.0, 10.1],
    }
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.kmeans(k=2, max_iterations=50)
    assert -1.0 <= r.silhouette_score <= 1.0


def test_kmeans_cluster_center_is_mean(tmp_path) -> None:
    data = {"a": [0.0, 0.0], "b": [2.0, 0.0], "c": [4.0, 0.0]}
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.kmeans(k=1, max_iterations=50)
    assert r.clusters[0].center[0] == pytest.approx(2.0, abs=0.5)


def test_kmeans_no_outliers_field(tmp_path) -> None:
    data = {"a": [1.0], "b": [2.0], "c": [3.0]}
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.kmeans(k=1)
    assert r.outliers == []


# ── dbscan ──────────────────────────────────────────────────────────────


def test_dbscan_empty(tmp_path) -> None:
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), {})
    r = s.dbscan()
    assert r.clusters == []
    assert r.total_memories == 0


def test_dbscan_all_outliers(tmp_path) -> None:
    data = {"a": [0.0], "b": [100.0], "c": [200.0]}
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.dbscan(eps=1.0, min_samples=3)
    assert r.total_memories == 3
    assert len(r.outliers) == 3
    assert r.clusters == []


def test_dbscan_one_cluster(tmp_path) -> None:
    data = {
        "a": [0.0, 0.0],
        "b": [0.1, 0.0],
        "c": [0.0, 0.1],
        "d": [0.1, 0.1],
    }
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.dbscan(eps=1.0, min_samples=2)
    assert r.total_memories == 4
    assert len(r.clusters) == 1
    assert r.clusters[0].size == 4
    assert r.outliers == []


def test_dbscan_two_clusters_with_noise(tmp_path) -> None:
    data = {
        "a1": [0.0, 0.0],
        "a2": [0.1, 0.0],
        "a3": [0.0, 0.1],
        "b1": [10.0, 10.0],
        "b2": [10.1, 10.0],
        "b3": [10.0, 10.1],
        "noise": [50.0, 50.0],
    }
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.dbscan(eps=1.0, min_samples=2)
    assert r.total_memories == 7
    assert len(r.clusters) == 2
    assert len(r.outliers) == 1
    assert r.outliers[0] == "noise"


def test_dbscan_silhouette_zero(tmp_path) -> None:
    data = {"a": [0.0], "b": [0.1]}
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.dbscan(eps=1.0, min_samples=2)
    assert r.silhouette_score == 0.0


def test_dbscan_min_samples_1(tmp_path) -> None:
    data = {"a": [0.0], "b": [100.0]}
    s = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    r = s.dbscan(eps=1.0, min_samples=1)
    assert r.total_memories == 2
    assert len(r.clusters) == 2
    assert r.outliers == []


# ── _calculate_silhouette ───────────────────────────────────────────────


def test_silhouette_empty_vectors(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    assert s._calculate_silhouette([], [], 0) == 0.0


def test_silhouette_zero_k(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    assert s._calculate_silhouette([[1.0]], [[]], 0) == 0.0


def test_silhouette_single_point_zero(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    v = s._calculate_silhouette([[1.0]], [[0]], 1)
    assert v == 0.0


def test_silhouette_perfect_clusters_high(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    vectors = [[0.0], [0.1], [10.0], [10.1]]
    clusters = [[0, 1], [2, 3]]
    sil = s._calculate_silhouette(vectors, clusters, 2)
    assert sil > 0.9


def test_silhouette_single_cluster_negative_one(tmp_path) -> None:
    # b_i guard: no other cluster → b_i=0; sil=(0-a)/a = -1.0
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    vectors = [[0.0], [10.0]]
    clusters = [[0, 1]]
    sil = s._calculate_silhouette(vectors, clusters, 1)
    assert sil == pytest.approx(-1.0)


# ── get_cluster_summary ─────────────────────────────────────────────────


def test_get_cluster_summary_keys(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    result = ClusteringResult(
        clusters=[
            Cluster(cluster_id=1, center=[0.5], members=["a", "b"], size=2, avg_distance=0.1),
        ],
        outliers=["noise1"],
        silhouette_score=0.75,
        total_memories=3,
    )
    summary = s.get_cluster_summary(result)
    assert summary["total_memories"] == 3
    assert summary["num_clusters"] == 1
    assert summary["num_outliers"] == 1
    assert summary["silhouette_score"] == 0.75
    assert len(summary["clusters"]) == 1
    assert summary["clusters"][0]["id"] == 1
    assert summary["clusters"][0]["size"] == 2
    assert summary["outliers"] == ["noise1"]


def test_get_cluster_summary_truncates_members(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    result = ClusteringResult(
        clusters=[
            Cluster(
                cluster_id=1,
                center=[0.0],
                members=["m1", "m2", "m3", "m4", "m5", "m6", "m7"],
                size=7,
                avg_distance=0.1,
            ),
        ],
        outliers=[],
        silhouette_score=0.0,
        total_memories=7,
    )
    summary = s.get_cluster_summary(result)
    assert len(summary["clusters"][0]["members"]) == 5


def test_get_cluster_summary_truncates_outliers(tmp_path) -> None:
    s = SemanticClusterer(embedding_path=str(tmp_path / "x"))
    outliers = [f"o{i}" for i in range(20)]
    result = ClusteringResult(clusters=[], outliers=outliers, silhouette_score=0.0, total_memories=20)
    summary = s.get_cluster_summary(result)
    assert len(summary["outliers"]) == 10


# ── get_clusterer singleton ─────────────────────────────────────────────


def test_get_clusterer_singleton(monkeypatch) -> None:
    monkeypatch.setattr(sc, "_clusterer", None)
    c1 = get_clusterer()
    c2 = get_clusterer()
    assert c1 is c2
    assert isinstance(c1, SemanticClusterer)


def test_get_clusterer_first_call_with_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sc, "_clusterer", None)
    p = str(tmp_path / "emb.json")
    c = get_clusterer(embedding_path=p)
    assert c._embedding_path == p


# ── cluster_memories ────────────────────────────────────────────────────


def test_cluster_memories_kmeans(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sc, "_clusterer", None)
    data = {
        "a1": [0.0],
        "a2": [0.1],
        "b1": [10.0],
        "b2": [10.1],
    }
    clusterer = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    monkeypatch.setattr(sc, "_clusterer", clusterer)
    summary = cluster_memories(k=2, method="kmeans")
    assert summary["num_clusters"] == 2
    assert summary["total_memories"] == 4


def test_cluster_memories_dbscan(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sc, "_clusterer", None)
    data = {"a1": [0.0], "a2": [0.1], "a3": [0.2]}
    clusterer = _inject(SemanticClusterer(embedding_path=str(tmp_path / "x")), data)
    monkeypatch.setattr(sc, "_clusterer", clusterer)
    summary = cluster_memories(k=2, method="dbscan")
    assert summary["total_memories"] == 3


def test_cluster_memories_unknown_method(monkeypatch) -> None:
    monkeypatch.setattr(sc, "_clusterer", None)
    result = cluster_memories(method="unknown")
    assert "error" in result
    assert "Unknown method" in result["error"]
