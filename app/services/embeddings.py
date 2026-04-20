"""
TradeMarkFlow-EU — 嵌入服务

使用 sentence-transformers 的 multilingual-e5-large 生成文本向量。
支持 100+ 种语言，单模型覆盖所有 24 种欧盟官方语言。

关键：query 前缀 "query: "，passage 前缀 "passage: " — E5 模型要求。
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Optional

import numpy as np

# ═══════════════════════════════════════════════════════════════
#  嵌入模型管理（单例，避免重复加载 ~2GB 模型）
# ═══════════════════════════════════════════════════════════════

_MODEL_NAME = "intfloat/multilingual-e5-large"
_model = None


def _get_model():
    """延迟加载模型 — 首次调用时加载（~30s），后续缓存。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedding] 加载模型: {_MODEL_NAME}...")
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"[Embedding] 模型就绪，维度: {_model.get_sentence_embedding_dimension()}")
    return _model


# ═══════════════════════════════════════════════════════════════
#  核心嵌入函数
# ═══════════════════════════════════════════════════════════════

def embed_query(text: str) -> list[float]:
    """
    为搜索查询生成嵌入向量。
    E5 模型要求 "query: " 前缀。
    """
    model = _get_model()
    prefixed = f"query: {text}"
    vec = model.encode(prefixed, normalize_embeddings=True)
    return vec.tolist()


def embed_passage(text: str) -> list[float]:
    """
    为存储的文档/商标生成嵌入向量。
    E5 模型要求 "passage: " 前缀。
    """
    model = _get_model()
    prefixed = f"passage: {text}"
    vec = model.encode(prefixed, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    批量嵌入，比逐条调用快 5-10x。

    Args:
        texts: 文本列表
        is_query: True 用 "query: " 前缀，False 用 "passage: "
    """
    model = _get_model()
    prefix = "query: " if is_query else "passage: "
    prefixed = [f"{prefix}{t}" for t in texts]
    vecs = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)
    return [v.tolist() for v in vecs]


def embed_trademark_name(mark_name: str, nice_classes: list[int] | None = None) -> list[float]:
    """
    为商标名称生成检索向量。

    策略：将尼斯分类信息拼入文本，让模型感知语义领域。
    例: "SolarFlux" + [9, 37] → "passage: SolarFlux | Class 9 Electronics; Class 37 Construction"
    """
    text = mark_name
    if nice_classes:
        class_str = ", ".join(f"Class {c}" for c in sorted(nice_classes))
        text = f"{mark_name} | {class_str}"
    return embed_passage(text)


def similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """余弦相似度（向量已归一化，所以 = 点积）。"""
    return float(np.dot(vec_a, vec_b))


# ═══════════════════════════════════════════════════════════════
#  缓存层（避免重复计算相同文本的嵌入）
# ═══════════════════════════════════════════════════════════════

_CACHE_DIR = "data/embedding_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_key(text: str, prefix: str) -> str:
    raw = f"{prefix}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def embed_with_cache(text: str, is_query: bool = False) -> list[float]:
    """
    带磁盘缓存的嵌入。相同文本只计算一次。
    适合商标数据入库时批量处理（避免重复 API 调用）。
    """
    prefix = "query" if is_query else "passage"
    key = _cache_key(text, prefix)
    cache_path = os.path.join(_CACHE_DIR, f"{key}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path).tolist()

    vec = embed_query(text) if is_query else embed_passage(text)
    np.save(cache_path, np.array(vec))
    return vec


# ═══════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("TradeMarkFlow-EU — 嵌入服务测试")
    print("=" * 60)

    # 测试多语言商标名
    test_marks = [
        ("SolarFlux", "English — solar energy"),
        ("SolaireFlux", "French — same concept, different language"),
        ("SonnenStrom", "German — 'sun power'"),
        ("СолнечныйПоток", "Russian — 'solar flow'"),
        ("EcoVolt", "English — unrelated mark"),
        ("Apple", "English — completely different domain"),
    ]

    print("\n[1] 生成嵌入向量...")
    t0 = time.time()
    vectors = {}
    for mark, desc in test_marks:
        vec = embed_passage(mark)
        vectors[mark] = vec
        print(f"  {mark:25s} dim={len(vec)}  [{desc}]")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # 测试跨语言相似度
    print("\n[2] 跨语言相似度矩阵:")
    query = embed_query("SolarFlux")
    print(f"  Query: 'SolarFlux' (query embedding)")
    print(f"  {'Mark':25s} {'Similarity':>10s}  评估")
    print(f"  {'-'*25} {'-'*10}  {'-'*20}")
    for mark, desc in test_marks:
        sim = similarity(query, vectors[mark])
        bar = "█" * int(sim * 20)
        label = "✓ 高相似" if sim > 0.85 else ("? 中等" if sim > 0.6 else "✗ 低相似")
        print(f"  {mark:25s} {sim:>10.4f}  {bar} {label}")

    print("\n✅ 嵌入服务测试完成")
