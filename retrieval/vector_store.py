"""VectorStore 抽象接口与多后端适配器。"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

# ★ 嵌入缓存命中率指标（失败时降级为 no-op，不影响主流程）
try:
    from omnimem.utils.metrics import record_cache_hit, record_cache_miss
except Exception:  # pragma: no cover - 监控模块不可用时降级
    def record_cache_hit() -> None:
        pass

    def record_cache_miss() -> None:
        pass

# ── 重型数值导入链预热 ──────────────────────────────────────────────────────
_PREHEAT_MODULES = ("numpy", "scipy.sparse", "sklearn.metrics", "chromadb")
_preheat_lock = threading.Lock()
_preheated = False


def preheat_heavy_imports() -> bool:
    """单线程把 numpy/scipy/sklearn/chromadb 的**首次**导入跑完，返回是否全部可用。

    ★ P1-6 追加：不这样做的话，ChromaDB 初始化线程和嵌入模型加载线程会各自第一次
    import 同一个数值图，而 numpy 的惰性子模块加载（``numpy/__init__.py`` 的
    ``__getattr__`` → ``import numpy.dtypes``）在「另一个线程正在执行 numpy 的
    ``__init__``」时会自我递归直到 RecursionError，并把 ``numpy._typing`` 留在
    **半初始化**状态写进 sys.modules。后果是全进程后续每一次数值导入都失败
    （实测表现：向量通道永久静默失效 + 日志谎报 "chromadb not installed"，
    线上待回填队列积压 6001 条查不到成因）。

    预热后这些导入退化为 sys.modules 命中，多线程不再有初始化窗口。冷启动代价
    约 1.4s（实测），常驻进程里为 0。
    """
    global _preheated
    if _preheated:
        return True
    with _preheat_lock:
        if _preheated:
            return True
        ok = True
        for name in _PREHEAT_MODULES:
            try:
                __import__(name)
            except Exception as e:
                # 可选依赖缺失是正常降级路径（BM25-only），不抛；但要说清是**真缺失**
                # 还是被并发导入污染，否则又会变成一条假消息。
                logging.getLogger(__name__).warning(
                    "预热 %s 失败（%s: %s）", name, type(e).__name__, e
                )
                ok = False
        _preheated = True
        return ok


# ★ 抑制 ChromaDB 0.6.x telemetry PostHog capture() 签名不兼容的噪音日志
#    这是 ChromaDB 内部问题，不影响功能，无需暴露给用户
class _ChromaDBTelemetryFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # 过滤 PostHog telemetry 失败消息和 chromadb telemetry DEBUG 噪音
        if "Failed to send telemetry event" in msg:
            return False
        if record.name.startswith("chromadb.telemetry"):
            return False
        return True

_telemetry_filter = _ChromaDBTelemetryFilter()
for _logger_name in ("chromadb.telemetry.product.posthog", "chromadb.telemetry"):
    logging.getLogger(_logger_name).addFilter(_telemetry_filter)
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _emit(msg: str) -> None:
    """向用户输出状态消息。

    优先写入 stderr（CLI 中可见），
    非 TTY 环境（gateway/cron/background）降级为 logger.info。
    """
    try:
        import sys
        if sys.stderr.isatty():
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        else:
            logger.info(msg)
    except Exception:
        logger.info(msg)


def load_embedding_cache_dict(path: str | Path) -> dict[str, list[float]]:
    """★ P2: 读取嵌入缓存为 {hash: vector} 字典。

    兼容两种持久化格式（供治理模块等外部消费方使用）：
      1. SQLite 新格式（<path去扩展名>.db，embedding_cache 表）— 优先
      2. JSON 旧格式（原 embedding_cache.json）— 回退
    """
    import json
    import os as _os
    import sqlite3

    p = Path(_os.path.expanduser(str(path)))
    db_path = p.with_suffix(".db")
    result: dict[str, list[float]] = {}

    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                rows = conn.execute("SELECT hash, vector FROM embedding_cache").fetchall()
            finally:
                conn.close()
            for key, vec_json in rows:
                try:
                    result[key] = [float(v) for v in json.loads(vec_json)]
                except Exception:
                    continue
            return result
        except Exception as e:
            logger.warning("load_embedding_cache_dict sqlite failed: %s", e)

    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            for key, vec in data.items():
                if isinstance(vec, (list, tuple)) and len(vec) == 2 and isinstance(vec[0], (list, tuple)):
                    result[key] = [float(v) for v in vec[0]]
                else:
                    result[key] = [float(v) for v in vec]
        except Exception as e:
            logger.warning("load_embedding_cache_dict json failed: %s", e)
    return result


# ★ P1-2: 待回填向量队列的文件名与路径推导必须只有一处定义。
#   写方（ChromaDBStore._record_unindexed）与读方（VectorRetriever.drain_vector_pending）
#   各自拼路径，是这条死信队列写了 6 001 条却永远没人读的原因之一。
VECTOR_PENDING_FILENAME = "vector_index_pending.jsonl"


def vector_pending_path(persist_dir: str | Path) -> Path:
    """待回填向量队列路径（向量库目录的父级，与历史写入位置保持一致）。"""
    return Path(persist_dir).parent / VECTOR_PENDING_FILENAME


def read_vector_pending(path: Path) -> list[dict[str, Any]]:
    """读取待回填队列。

    逐行 try/except（同 governance/sync.py 的 jsonl 读法）：单行损坏只跳过，不拖垮整批。
    按 id 去重保留最后一条 —— 同一条记忆可能被多次降级写入，最新内容才是对的。
    """
    import json

    entries: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    logger.debug("跳过损坏的待回填向量行: %s", line[:120])
                    continue
                if not isinstance(record, dict):
                    continue
                doc_id = str(record.get("id") or "")
                if not doc_id or not record.get("document"):
                    continue
                entries[doc_id] = record
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning("读取待回填向量队列失败: %s", e)
        return []
    return list(entries.values())


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        ...

    @abstractmethod
    def query(self, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class _CachedEmbeddingFunction:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_path: Path | None = None,
        model_path: str = "",
        cache_ttl: float = 3600,
        load_timeout: float = 90.0,
    ):
        self._model_name = model_name
        self._model_path = model_path
        self._model = None
        # ★ 模型就绪事件：加载结束时置位（成功或失败都要置位，见 _get_model 的 finally）
        self._model_ready = threading.Event()
        # ★ P1-6：等待者不能因为加载线程异常退出而永久挂起，所以失败也要显式记录
        self._load_timeout = load_timeout
        self._load_lock = threading.Lock()
        self._load_lock_wait = 5.0
        self._load_retry_cooldown = 60.0
        self._load_thread: threading.Thread | None = None
        self._load_box: dict[str, Any] = {}
        self._load_error: BaseException | None = None
        self._load_failed_at = 0.0
        # ★ TTL + LRU 缓存：value 为 (embedding, expire_at)
        self._cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._max_cache = 1000
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._cache_path = cache_path
        # ★ 异步持久化：脏标记 + 持久化进行中标记
        self._dirty = False
        self._persist_in_progress = False
        self._load_cache()

    @staticmethod
    def name() -> str:
        """ChromaDB EmbeddingFunction 协议要求的 name 方法。"""
        return "omnimem_cached_sentence_transformer"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> _CachedEmbeddingFunction:
        """ChromaDB EmbeddingFunction 协议要求的反序列化方法。"""
        return _CachedEmbeddingFunction(
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
            cache_path=Path(config["cache_path"]) if config.get("cache_path") else None,
            # Without model_path a rebuilt function resolves model_name against the Hub instead of
            # the local snapshot, and silently embeds with a different model into a 384-dim space.
            model_path=config.get("model_path", ""),
        )

    def get_config(self) -> dict[str, Any]:
        """ChromaDB EmbeddingFunction 协议要求的序列化方法。"""
        return {
            "model_name": self._model_name,
            "model_path": self._model_path,
            "cache_path": str(self._cache_path) if self._cache_path else "",
        }

    @staticmethod
    def is_legacy() -> bool:
        return False

    @property
    def cache_size(self) -> int:
        """当前内存缓存条目数。"""
        with self._lock:
            return len(self._cache)

    def _db_path(self) -> Path | None:
        """★ P2: SQLite 缓存文件路径（与旧 JSON 同目录，扩展名 .db）。"""
        if not self._cache_path:
            return None
        return self._cache_path.with_suffix(".db")

    @staticmethod
    def _open_db(db_path: Path) -> Any:
        """打开 SQLite 缓存连接（每次操作独立连接，天然线程安全）。"""
        import sqlite3

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache ("
            "hash TEXT PRIMARY KEY, vector TEXT NOT NULL, updated_at REAL)"
        )
        return conn

    # SQLite 缓存最大行数（磁盘层，大于内存层 _max_cache）
    _DB_MAX_ROWS = 5000

    def _load_cache(self) -> None:
        """★ P2: 从 SQLite 加载缓存；首次运行时自动导入旧 JSON 格式。"""
        db_path = self._db_path()
        if db_path is None:
            return
        import json

        try:
            # 旧 JSON 存在且 SQLite 尚未建立 → 一次性导入（JSON 保留不再写入）
            if not db_path.exists() and self._cache_path.exists():
                self._migrate_json_to_db(db_path)

            if not db_path.exists():
                return
            conn = self._open_db(db_path)
            try:
                rows = conn.execute(
                    "SELECT hash, vector FROM embedding_cache "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (self._max_cache,),
                ).fetchall()
            finally:
                conn.close()
            now = time.time()
            loaded: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
            # 按 updated_at 升序放入（最新的最后 → LRU 末端）
            for key, vec_json in reversed(rows):
                try:
                    loaded[key] = ([float(v) for v in json.loads(vec_json)], now + self._cache_ttl)
                except Exception:
                    continue
            self._cache = loaded
            count = len(self._cache)
            logger.warning("Loaded %d entries from embedding cache (sqlite)", count)
            if count > 0:
                _emit(f"[OmniMem] 嵌入缓存: 从磁盘加载 {count} 条")
        except Exception as e:
            logger.warning("Embedding cache load failed: %s", e)
            self._cache = OrderedDict()

    def _migrate_json_to_db(self, db_path: Path) -> None:
        """旧 JSON 缓存一次性导入 SQLite（导入失败不阻断启动）。"""
        import json

        try:
            with open(self._cache_path, encoding="utf-8") as f:
                data = json.load(f)
            now = time.time()
            rows = []
            for k, vec in data.items():
                # 新格式 (vector, expire_at) 的判据是首元素为列表，
                # 避免把长度恰为 2 的旧格式向量误判为元组格式
                if (
                    isinstance(vec, (list, tuple))
                    and len(vec) == 2
                    and isinstance(vec[0], (list, tuple))
                ):
                    embedding = vec[0]
                else:
                    embedding = vec  # 旧格式: 仅 vector
                rows.append((k, json.dumps([float(v) for v in embedding]), now))
            conn = self._open_db(db_path)
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO embedding_cache (hash, vector, updated_at) "
                    "VALUES (?, ?, ?)",
                    rows,
                )
                conn.commit()
            finally:
                conn.close()
            logger.warning("Migrated %d embedding cache entries JSON -> SQLite", len(rows))
        except Exception as e:
            logger.warning("Embedding cache JSON->SQLite migration failed: %s", e)

    def persist(self) -> None:
        """异步持久化嵌入缓存到磁盘（后台线程写入，不阻塞主流程）。

        使用 _dirty 标记避免无变更时的无效写入，
        使用 _persist_in_progress 标记避免并发写入冲突。

        ★ 修复 C12：原实现 _dirty / _persist_in_progress 检查-赋值非原子，
           多线程可能同时通过检查并启动多个持久化线程。改为持锁检查并设置标志。
        """
        if not self._cache_path:
            return
        with self._lock:
            # 无变更则跳过
            if not self._dirty:
                return
            # 已有持久化在进行中则跳过（避免并发写入冲突）
            if self._persist_in_progress:
                return
            # 标记持久化进行中（持锁，原子操作）
            self._persist_in_progress = True
        # 启动后台 daemon 线程执行磁盘写入（锁外执行 IO，避免长持锁）
        thread = threading.Thread(target=self._persist_to_disk, daemon=True)
        thread.start()

    def _persist_to_disk(self) -> None:
        """后台线程执行磁盘写入（★ P2: SQLite 增量 upsert，替代 JSON 全量重写）。"""
        try:
            import json

            db_path = self._db_path()
            if db_path is None:
                return
            with self._lock:
                # 快照当前内存缓存（仅向量，加载时重置 TTL）
                data = {k: v[0] for k, v in self._cache.items()}
                self._dirty = False
            now = time.time()
            conn = self._open_db(db_path)
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO embedding_cache (hash, vector, updated_at) "
                    "VALUES (?, ?, ?)",
                    [(k, json.dumps(v), now) for k, v in data.items()],
                )
                # 磁盘层容量裁剪：保留最近使用的 _DB_MAX_ROWS 行
                conn.execute(
                    "DELETE FROM embedding_cache WHERE hash NOT IN ("
                    "SELECT hash FROM embedding_cache ORDER BY updated_at DESC LIMIT ?)",
                    (self._DB_MAX_ROWS,),
                )
                conn.commit()
            finally:
                conn.close()
            logger.warning("Saved %d entries to embedding cache (sqlite, async)", len(data))
        except Exception as e:
            logger.warning("Embedding cache persist failed: %s", e)
        finally:
            self._persist_in_progress = False

    def _adopt_inflight_load(self) -> bool:
        """收取后台加载线程的结果。

        加载超时只是「这次不想再等」，线程往往随后还是会成功 —— 下次调用把成果捡回来，
        不要重复起第二个 12s 的加载。
        """
        box = self._load_box
        if "result" in box:
            self._model = box["result"]
            self._load_error = None
            return True
        if "error" in box and self._load_error is None:
            self._load_error = box["error"]
            self._load_failed_at = time.monotonic()
        return False

    def _get_model(self) -> Any:
        """返回可用的嵌入模型；不可用时**抛异常**而不是挂住等待者。

        ★ P1-6：本方法原先把 ``_model_ready.set()`` 放在最内层 try 的 finally 里，
        于是 ``from sentence_transformers import ...`` 之类的早期失败永远不会置位，
        所有 ``wait_ready()`` 调用方只能各自等到超时；多个线程同时进来还会各起一次
        完整模型加载。现在：单加载线程 + 有界 join + 任何退出路径都置位。
        """
        if self._model is not None:
            return self._model
        try:
            if self._adopt_inflight_load() and self._model is not None:
                return self._model
            if self._load_error is not None:
                # 冷却期内快速失败（走 BM25+FTS5 降级），过期后允许重试
                if time.monotonic() - self._load_failed_at < self._load_retry_cooldown:
                    raise self._load_error
                self._load_error = None
            if not self._load_lock.acquire(timeout=self._load_lock_wait):
                raise TimeoutError(
                    f"等待嵌入模型加载锁超过 {self._load_lock_wait}s（已有其他线程在加载）"
                )
            try:
                if self._model is None:
                    self._load_model_locked()
            finally:
                self._load_lock.release()
            if self._model is None:
                raise self._load_error or RuntimeError("嵌入模型不可用")
            return self._model
        finally:
            # ★ 无论成功、失败还是超时，等待者都必须被唤醒
            self._model_ready.set()

    def _load_model_locked(self) -> None:
        import os

        if self._load_thread is not None and self._load_thread.is_alive():
            raise TimeoutError("嵌入模型仍在后台加载中，不重复启动第二次加载")

        # ★ 仅当 skill-router 模型路径与本模型一致时才复用（避免维度不匹配）
        #   skill-router 使用 768 维微调 BERT，OmniMem 使用 384 维 MiniLM，
        #   维度不同会导致 FAISS 索引完全失效。
        try:
            import sys
            sr = sys.modules.get("hermes_plugins.skill_router")
            if sr is None:
                import importlib
                sr = importlib.import_module("hermes_plugins.skill_router")
            cached = getattr(sr, "_MODEL_CACHE", None)
            if cached is not None:
                # 检查模型路径是否一致（resolve 绝对路径比较）
                sr_model_path = os.path.expanduser(
                    sr._load_config().get("model_path", "") if hasattr(sr, "_load_config") else ""
                )
                my_model_path = os.path.abspath(self._model_path or self._model_name)
                if sr_model_path and os.path.abspath(sr_model_path) == my_model_path:
                    self._model = cached
                    logger.info("[OmniMem] 复用 skill-router 的 embedding 模型（路径一致）")
                    return
                else:
                    logger.debug("[OmniMem] 跳过 skill-router 模型复用（路径不同: sr=%s, omni=%s）",
                               sr_model_path, my_model_path)
        except Exception:
            logger.debug("OmniMem skill-router model reuse check failed", exc_info=True)

        model_path = self._model_path or self._model_name
        _emit(f"[OmniMem] 正在加载嵌入模型: {model_path} ...")
        t0 = time.time()

        # SentenceTransformer(...) 本身不可中断，所以放进守护线程再对它设一个界；
        # 超时后由 _adopt_inflight_load 在后续调用里回收结果。
        box: dict[str, Any] = {}
        self._load_box = box

        def _work() -> None:
            # ★ P1-6 追加：整条重度导入链要跑在这个新线程的浅栈上，而不是调用方栈上。
            #   sentence_transformers → sklearn → scipy → torch 的导入链本身就有数百层，
            #   调用方常常已经是 chromadb add() + saga + middleware 的深栈，两者相加撑爆
            #   默认 recursionlimit(1000)，表现即写入时 "maximum recursion depth exceeded
            #   in add"、整批向量索引被跳过。旧代码用 sys.setrecursionlimit(5000) 掩盖成因
            #   （P2-3a 已删），这里换成真正的解法。
            try:
                # 这些开关必须先于 import sentence_transformers：huggingface_hub 在导入时
                # 就把 HF_ENDPOINT / HF_HUB_OFFLINE 冻结进自己的 constants，之后再改无效
                # —— 表现是仍去连 huggingface.co，连不上又报"找不到缓存文件"，于是每次
                # 记忆写入都跳过向量索引（2026-09-27/28 实况）。
                if "HF_ENDPOINT" not in os.environ:
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # GFW 镜像
                # 冷启动加速：本地模型路径存在时强制离线，跳过 HF 联网检查
                # （本地 MiniLM 加载 ~12s，联网检查额外 ~14s，甚至超时）
                if self._model_path and os.path.isdir(self._model_path):
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

                try:
                    import torch.distributed as dist

                    if not hasattr(dist, "is_initialized"):
                        dist.is_initialized = lambda: False
                except Exception:
                    logger.debug("OmniMem torch.distributed patch failed", exc_info=True)

                from sentence_transformers import SentenceTransformer

                # 设备自动检测: bge-m3 等大模型在 CPU 上编码会超时触发熔断
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
                box["device"] = device
                box["result"] = SentenceTransformer(model_path, device=device)
            except BaseException as exc:  # noqa: BLE001 - 需要把任何失败带回主线程
                box["error"] = exc

        thread = threading.Thread(target=_work, name="omnimem-embed-load", daemon=True)
        self._load_thread = thread
        thread.start()
        thread.join(self._load_timeout)

        if thread.is_alive():
            self._load_error = TimeoutError(
                f"嵌入模型加载超过 {self._load_timeout}s: {model_path}"
            )
            self._load_failed_at = time.monotonic()
            _emit(f"[OmniMem] ⚠ 嵌入模型加载超时 ({self._load_timeout}s)，本次降级为关键词检索")
            raise self._load_error
        if "error" in box:
            exc = box["error"]
            self._load_error = exc
            self._load_failed_at = time.monotonic()
            _emit(f"[OmniMem] ⚠ 嵌入模型加载失败: {exc}")
            raise exc
        self._model = box["result"]
        self._load_error = None
        elapsed = time.time() - t0
        _emit(f"[OmniMem] 嵌入模型加载完成 ({model_path}, {elapsed:.1f}s, {box.get('device', '?').upper()})")

    def wait_ready(self, timeout: float = 30.0) -> bool:
        """等待模型加载完成（含首次加载），返回是否就绪。

        Args:
            timeout: 最大等待秒数（首次加载约 14s，默认 30s 充裕）

        Returns:
            True 表示模型可用；False 表示超时/不可用
        """
        if self._model is not None:
            return True
        self._model_ready.wait(timeout=timeout)
        self._adopt_inflight_load()
        return self._model is not None

    def ensure_loaded(self) -> bool:
        """同步确保模型已加载（若未加载则立即触发加载）。"""
        try:
            self._get_model()
            return self._model is not None
        except Exception:
            return False

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_query(input)

    def _evict_expired(self, now: float) -> None:
        """清理已过期的缓存条目。"""
        expired = [k for k, (_, expire_at) in self._cache.items() if expire_at <= now]
        for k in expired:
            del self._cache[k]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        results = []
        to_encode = []
        to_encode_idx = []
        now = time.time()

        with self._lock:
            for i, text in enumerate(input):
                text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                cached = self._cache.get(text_hash)
                if cached is not None:
                    vec, expire_at = cached
                    if expire_at > now:
                        # ★ 缓存命中且未过期：更新 LRU 顺序并记录指标
                        self._cache.move_to_end(text_hash)
                        results.append((i, vec))
                        try:
                            record_cache_hit()
                        except Exception:
                            pass
                        continue
                    # 已过期则删除，重新编码
                    del self._cache[text_hash]
                # ★ 缓存未命中：记录指标
                to_encode.append(text)
                to_encode_idx.append((i, text_hash))
                try:
                    record_cache_miss()
                except Exception:
                    pass

        if to_encode:
            model = self._get_model()
            embeddings = model.encode(to_encode, convert_to_numpy=True)
            now = time.time()
            with self._lock:
                for (orig_idx, text_hash), emb in zip(to_encode_idx, embeddings, strict=False):
                    vec = emb.tolist()
                    self._cache[text_hash] = (vec, now + self._cache_ttl)
                    self._cache.move_to_end(text_hash)
                    results.append((orig_idx, vec))

                # ★ TTL 优先淘汰，再按 LRU 容量淘汰
                self._evict_expired(now)
                while len(self._cache) > self._max_cache:
                    self._cache.popitem(last=False)
                # ★ 缓存已变更，标记为脏（下次 persist 时写入磁盘）
                self._dirty = True

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


class ChromaDBStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "omnimem",
        persist_dir: str | Path = "/tmp/omnimem/retrieval/chroma",
        embedding_fn: Any = None,
    ):
        self._collection_name = collection_name
        self._persist_dir = Path(persist_dir)
        self._embedding_fn = embedding_fn
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False
        self._persist_pending = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            # ★ 只有这一步的 ImportError 才真的等价于「chromadb 没装」
            logger.warning("chromadb 不可用（ImportError: %s）— 向量检索降级", e)
            self._initialized = True
            return
        try:
            # ★ 并发锁修复：SQLite busy_timeout=10s + WAL 模式，避免 Gateway 常驻进程
            #   与短命脚本进程同时访问 ChromaDB 时 database is locked 错误。
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            # ChromaDB 0.6.x — 通过 sqlite_pragma 启用 WAL + busy_timeout
            import sqlite3
            _db_path = str(self._persist_dir / "chroma.sqlite3")
            if __import__("os").path.isfile(_db_path):
                try:
                    _conn = sqlite3.connect(_db_path, timeout=10.0)
                    _conn.execute("PRAGMA journal_mode=WAL")
                    _conn.execute("PRAGMA busy_timeout=10000")
                    _conn.close()
                except Exception:
                    pass
            # ★ R32修复QUAL-2：ChromaDB 0.6.x 升级后 config_json_str 缺少 _type 字段
            # 旧版 ChromaDB 创建的 collection 配置为 {}，新版期望 {"_type": "CollectionConfigurationInternal"}
            # 在获取 collection 前自动修复
            self._fix_chromadb_config_type()
            try:
                if self._embedding_fn is not None:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                        embedding_function=self._embedding_fn,
                    )
                else:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
            except Exception as e:
                # embedding function 不兼容时，删除旧 collection 重建
                logger.warning("ChromaDB collection incompatible: %s, recreating", e)
                try:
                    self._client.delete_collection(name=self._collection_name)
                except Exception:
                    logger.debug("OmniMem ChromaDB collection delete failed during recreate", exc_info=True)
                if self._embedding_fn is not None:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                        embedding_function=self._embedding_fn,
                    )
                else:
                    self._collection = self._client.get_or_create_collection(
                        name=self._collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
            logger.warning(
                "ChromaDB collection initialized: %d documents",
                self._collection.count() if self._collection else 0,
            )
        except Exception as e:
            # ★ 诊断修复：原先把 ImportError 一律报成 "chromadb not installed"，但这个
            #   try 里任何**嵌套**导入失败（chromadb 的可选依赖、被中断而半初始化的
            #   numpy）都会落到那条假消息上，真实成因被抹掉 —— 线上「向量通道静默失效
            #   + 待回填积压 6001 条」正是这么丢线索的。现在保留异常类型与堆栈。
            logger.warning(
                "ChromaDB init failed: %s: %s", type(e).__name__, e, exc_info=True
            )
        self._initialized = True

    def _fix_chromadb_config_type(self) -> None:
        """★ R32修复QUAL-2：修复 ChromaDB 0.6.x 升级后 config_json_str 缺少 _type 字段的问题。

        ChromaDB 0.6.x 的 CollectionConfigurationInternal.from_json() 要求
        config_json_str 中包含 "_type" 字段，但旧版创建的 collection
        config_json_str 为 {}，导致 KeyError: '_type'。
        """
        try:
            import sqlite3

            chroma_db_path = self._persist_dir / "chroma.sqlite3"
            if not chroma_db_path.exists():
                return
            conn = sqlite3.connect(str(chroma_db_path))
            try:
                rows = conn.execute(
                    "SELECT name, config_json_str FROM collections"
                ).fetchall()
                for name, config_str in rows:
                    if not config_str:
                        continue
                    try:
                        import json

                        config = json.loads(config_str)
                        if "_type" not in config:
                            config["_type"] = "CollectionConfigurationInternal"
                            conn.execute(
                                "UPDATE collections SET config_json_str = ? WHERE name = ?",
                                (json.dumps(config), name),
                            )
                            logger.info(
                                "ChromaDB config migrated: added _type to collection %s",
                                name,
                            )
                    except (json.JSONDecodeError, Exception):
                        logger.debug("OmniMem ChromaDB config migration row skipped", exc_info=True)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("ChromaDB config migration skipped: %s", e)

    def pending_path(self) -> Path:
        """★ P1-2: 本后端的待回填队列路径。写方与读方都走这里，不再各拼一次路径。"""
        return vector_pending_path(self._persist_dir)

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        self._ensure_initialized()
        if self._collection is None:
            logger.error(
                "ChromaDBStore unavailable (init failed earlier); SKIPPED vector indexing for %d doc(s): %s",
                len(ids), ids,
            )
            self._record_unindexed(ids, documents, metadatas, reason="collection_unavailable")
            return
        # ★ P0-1 修复：三段式降级，防止 ChromaDB upsert 递归溢出阻塞主流程
        # 阶段 1: 尝试 upsert（覆盖已存在的）
        try:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            self._persist_client()
            return
        except RecursionError as e:
            logger.warning("ChromaDBStore upsert hit RecursionError, fallback to add: %s", e)
        except Exception as e:
            logger.warning("ChromaDBStore upsert failed, fallback to add: %s", e)

        # 阶段 2: 降级为 add（不覆盖已存在的）
        try:
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
            self._persist_client()
            return
        except Exception as e:
            # 终态错误必须自带 embedder 身份：_get_model 的 "[OmniMem] 正在加载嵌入模型" 走
            # _emit，某些进程里不进 agent.log，只有这一行能证明用的是本地快照还是 Hub 模型名。
            _embedder = (
                getattr(self._embedding_fn, "_model_path", "")
                or getattr(self._embedding_fn, "_model_name", "?")
            )
            logger.error(
                "ChromaDBStore add fallback also failed; SKIPPED vector indexing for %d doc(s). "
                "Semantic recall will miss these until backfilled. cause=%s embedder=%s",
                len(ids), e, _embedder, exc_info=True,
            )
            self._record_unindexed(ids, documents, metadatas, reason="embed_or_write_failed")

        # 阶段 3: 跳过向量索引（主存储 SQLite 不受影响）
        # 不抛异常，让主流程继续；但必须留下可回填的队列，见 _record_unindexed。

    def _record_unindexed(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None,
        reason: str,
    ) -> None:
        """Append skipped docs to a backfill queue so a silent index loss stays recoverable.

        Never raises: a failure here must not propagate into the memory write path.
        """
        import json

        try:
            queue = vector_pending_path(self._persist_dir)
            queue.parent.mkdir(parents=True, exist_ok=True)
            with queue.open("a", encoding="utf-8") as fh:
                for i, doc_id in enumerate(ids):
                    fh.write(json.dumps({
                        "ts": time.time(),
                        "id": doc_id,
                        "document": documents[i] if i < len(documents) else "",
                        "metadata": (metadatas[i] if metadatas and i < len(metadatas) else {}),
                        "reason": reason,
                    }, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("ChromaDBStore could not record unindexed backlog for %d doc(s)", len(ids), exc_info=True)

    def query(self, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
        self._ensure_initialized()
        if self._collection is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        try:
            count = self._collection.count()
            if count == 0:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            kwargs: dict[str, Any] = {
                "query_texts": query_texts,
                "n_results": min(n_results, count),
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                kwargs["where"] = where
            return self._collection.query(**kwargs)
        except Exception as e:
            logger.warning("ChromaDBStore query failed: %s", e)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete(self, ids: list[str]) -> None:
        self._ensure_initialized()
        if self._collection is None:
            return
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
            self._persist_client()
        except Exception as e:
            logger.warning("ChromaDBStore delete failed: %s", e)

    def count(self) -> int:
        self._ensure_initialized()
        if self._collection is None:
            return 0
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def all_ids(self) -> list[str]:
        """向量库里的全部原始 id（chunk 级）。★ P1-11：覆盖率判据需要它，`count()` 不够。"""
        self._ensure_initialized()
        if self._collection is None:
            return []
        try:
            return [str(i) for i in self._collection.get(include=[])["ids"]]
        except Exception:
            try:
                data = self._collection.get()
                return [str(i) for i in data["ids"]]
            except Exception as e:
                logger.warning("ChromaDBStore all_ids failed: %s", e)
                return []

    def reset(self) -> None:
        self._ensure_initialized()
        if self._client is None:
            return
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_fn,
            )
        except Exception as e:
            logger.warning("ChromaDBStore reset failed: %s", e)

    def _persist_client(self) -> None:
        try:
            if self._client and hasattr(self._client, "persist"):
                self._client.persist()
                self._persist_pending = False
        except Exception as e:
            logger.warning("ChromaDB persist failed: %s", e)
            self._persist_pending = True

    def close(self) -> None:
        """释放 ChromaDB 客户端和集合。"""
        try:
            self._persist_client()
            self._collection = None
            self._client = None
            self._initialized = False
        except Exception as e:
            logger.warning("ChromaDBStore close failed: %s", e)


class QdrantStore(VectorStore):
    """Qdrant 向量存储后端。

    ★ P0修复：接入真实 embedding 向量（原实现使用全零向量，检索完全失效）。
      - embedding_fn 缺失时 add/query 显式抛出 RuntimeError，不再静默降级
      - point id 使用 memory_id 派生的确定性 UUID，重启后 upsert 幂等
    """

    def __init__(
        self,
        collection_name: str = "omnimem",
        url: str | None = None,
        api_key: str | None = None,
        embedding_fn: _CachedEmbeddingFunction | None = None,
        dimension: int = 384,
    ):
        self._collection_name = collection_name
        self._url = url or "localhost:6333"
        self._api_key = api_key
        self._embedding_fn = embedding_fn
        self._dimension = dimension
        self._client: Any = None
        self._initialized = False

    @staticmethod
    def _point_id(memory_id: str) -> str:
        """memory_id → 确定性 UUID（Qdrant 要求 point id 为整数或 UUID）。"""
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"omnimem:{memory_id}"))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """计算真实 embedding。缺失 embedding_fn 时显式报错，不静默降级。"""
        if self._embedding_fn is None:
            raise RuntimeError(
                "QdrantStore requires an embedding_fn — vector search would be "
                "broken with dummy vectors. Install sentence-transformers or "
                "pass embedding_fn explicitly."
            )
        return self._embedding_fn.embed_query(texts)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            kwargs: dict[str, Any] = {"url": self._url}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = QdrantClient(**kwargs)
            try:
                self._client.get_collection(self._collection_name)
            except Exception:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
                )
            logger.warning("Qdrant collection initialized: %s", self._collection_name)
        except ImportError:
            logger.warning("qdrant-client not installed — Qdrant vector search unavailable")
        except Exception as e:
            logger.warning("Qdrant init failed: %s", e)
        self._initialized = True

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] | None = None) -> None:
        self._ensure_initialized()
        if self._client is None:
            return
        # ★ 真实向量计算：embedding 失败应显式暴露而非静默写入坏数据
        vectors = self._embed(documents)
        try:
            from qdrant_client.models import PointStruct

            metas = metadatas or [{}] * len(ids)
            points = []
            for pid, doc, meta, vec in zip(ids, documents, metas, vectors, strict=False):
                points.append(
                    PointStruct(
                        id=self._point_id(pid),
                        vector=vec,
                        payload={"_id": pid, "document": doc, **meta},
                    )
                )
            self._client.upsert(collection_name=self._collection_name, points=points)
        except Exception as e:
            logger.warning("QdrantStore add failed: %s", e)

    def query(self, query_texts: list[str], n_results: int = 10, where: dict | None = None) -> dict:
        self._ensure_initialized()
        if self._client is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        # ★ 真实查询向量（原实现使用全零 dummy_vector，结果与语义无关）
        query_vector = self._embed([query_texts[0]])[0] if query_texts else [0.0] * self._dimension
        try:
            filter_obj = None
            if where:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                conditions = []
                for k, v in where.items():
                    conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
                filter_obj = Filter(must=conditions)

            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=n_results,
                query_filter=filter_obj,
            )
            # ★ 返回 payload 中的原始 memory_id（而非 Qdrant 内部 point id）
            ids = [[str(r.payload.get("_id", r.id)) for r in results]]
            documents = [[r.payload.get("document", "") for r in results]]
            metadatas = [[{k: v for k, v in r.payload.items() if k != "document"} for r in results]]
            distances = [[1.0 - r.score for r in results]]
            return {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
            }
        except Exception as e:
            logger.warning("QdrantStore query failed: %s", e)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete(self, ids: list[str]) -> None:
        self._ensure_initialized()
        if self._client is None:
            return
        try:
            from qdrant_client.models import PointIdsList

            # ★ point id 为 memory_id 的确定性 UUID，可直接定位，无需 scroll 全表
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=[self._point_id(i) for i in ids]),
            )
        except Exception as e:
            logger.warning("QdrantStore delete failed: %s", e)

    def count(self) -> int:
        self._ensure_initialized()
        if self._client is None:
            return 0
        try:
            info = self._client.get_collection(self._collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def reset(self) -> None:
        self._ensure_initialized()
        if self._client is None:
            return
        try:
            self._client.delete_collection(self._collection_name)
            from qdrant_client.models import Distance, VectorParams

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )
        except Exception as e:
            logger.warning("QdrantStore reset failed: %s", e)
