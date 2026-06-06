# -*- coding: utf-8 -*-
"""
LSA/LSI 与 LDA 主题建模。
理论部分用 numpy SVD；工程部分用 sklearn TruncatedSVD / LatentDirichletAllocation。
"""
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD


def numpy_svd_demo(tfidf_dense, k=5):
    """演示 SVD 低秩逼近：A ≈ U_k Σ_k V_k^T。"""
    U, s, Vt = np.linalg.svd(tfidf_dense, full_matrices=False)
    k = min(k, len(s))
    approx = (U[:, :k] * s[:k]) @ Vt[:k, :]
    energy = (s[:k] ** 2).sum() / (s ** 2).sum()
    return {
        "singular_values": s,
        "top_k_energy_ratio": float(energy),
        "approximation": approx,
        "U_k": U[:, :k],
        "Sigma_k": np.diag(s[:k]),
        "Vt_k": Vt[:k, :],
    }


def train_lsi(tfidf_matrix, n_components=5, random_state=42):
    model = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
    )
    doc_topics = model.fit_transform(tfidf_matrix)
    return model, doc_topics


def train_lda(count_matrix, n_topics=5, max_iter=30, random_state=42):
    model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method="batch",
        random_state=random_state,
    )
    doc_topics = model.fit_transform(count_matrix)
    return model, doc_topics


def top_terms_per_topic(model, feature_names, n_words=10, kind="lsi"):
    topics = []
    if kind == "lsi":
        components = model.components_
    else:
        components = model.components_
    for i, comp in enumerate(components):
        idx = np.argsort(-comp)[:n_words]
        words = [(feature_names[j], float(comp[j])) for j in idx]
        topics.append({"topic_id": i, "terms": words})
    return topics


def assign_dominant_topic(doc_topics):
    return np.argmax(doc_topics, axis=1)
