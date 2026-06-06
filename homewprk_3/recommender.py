# -*- coding: utf-8 -*-
"""基于降维向量的 KNN 文档推荐。"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_index(embeddings, metric="cosine", n_neighbors=6):
    nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)
    nn.fit(embeddings)
    return nn


def recommend(nn_model, embeddings, query_idx, k=5):
    vec = embeddings[query_idx : query_idx + 1]
    dists, idxs = nn_model.kneighbors(vec, n_neighbors=k + 1)
    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        if idx == query_idx:
            continue
        results.append({"doc_id": int(idx), "distance": float(dist)})
        if len(results) >= k:
            break
    return results


def query_by_keywords(tfidf_matrix, vectorizer, keywords, topn=5):
    q = vectorizer.transform([" ".join(keywords)])
    sims = (tfidf_matrix @ q.T).toarray().ravel()
    order = np.argsort(-sims)
    return [{"doc_id": int(i), "score": float(sims[i])} for i in order[:topn]]
