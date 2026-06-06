# -*- coding: utf-8 -*-
"""PCA / t-SNE / UMAP 可视化。"""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap

    _HAS_UMAP = True
except ImportError:
    umap = None
    _HAS_UMAP = False

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _color_map(labels):
    uniq = sorted(set(labels))
    cmap = plt.colormaps["tab10"].resampled(max(len(uniq), 1))
    lut = {lb: cmap(i % 10) for i, lb in enumerate(uniq)}
    return [lut[lb] for lb in labels], uniq


def reduce_2d(embeddings, method="pca", random_state=42, umap_neighbors=8, umap_min_dist=0.25):
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(embeddings)
    if method == "tsne":
        perp = min(30, max(5, embeddings.shape[0] // 4))
        return TSNE(
            n_components=2,
            perplexity=perp,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        ).fit_transform(embeddings)
    if method == "umap":
        if not _HAS_UMAP:
            raise RuntimeError("umap-learn 未安装")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(umap_neighbors, embeddings.shape[0] - 1),
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)
    raise ValueError(f"未知降维方法: {method}")


def plot_embedding_scatter(
    coords,
    labels,
    title,
    out_path,
    annotate=False,
    max_annotate=12,
):
    colors, uniq = _color_map(labels)
    fig, ax = plt.subplots(figsize=(9, 6))
    for lb in uniq:
        mask = [l == lb for l in labels]
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(xs, ys, label=lb, alpha=0.75, s=36)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    if annotate:
        for i in range(min(max_annotate, len(coords))):
            ax.annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=7)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_topic_terms(topic_list, out_path, n_topics=5):
    fig, axes = plt.subplots(1, n_topics, figsize=(3 * n_topics, 4))
    if n_topics == 1:
        axes = [axes]
    for ax, topic in zip(axes, topic_list[:n_topics]):
        words = [w for w, _ in topic["terms"][:8]]
        weights = [s for _, s in topic["terms"][:8]]
        ax.barh(words[::-1], weights[::-1], color="#4C72B0")
        ax.set_title(f"Topic {topic['topic_id']}")
    fig.suptitle("主题高频词")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_singular_values(singular_values, out_path, k=10):
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = singular_values[:k]
    ax.plot(range(1, len(vals) + 1), vals, marker="o")
    ax.set_xlabel("奇异值序号")
    ax.set_ylabel("奇异值大小")
    ax.set_title("SVD 奇异值谱（LSA/LSI 理论基础）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
