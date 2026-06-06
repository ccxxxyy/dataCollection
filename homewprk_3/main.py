# -*- coding: utf-8 -*-
"""作业三主流程：数据 → 降维 → 主题 → 可视化 → KNN 推荐。"""
import json
import os

import numpy as np

from config import (
    KNN_K,
    LDA_MAX_ITER,
    LSA_COMPONENTS,
    N_TOPICS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TSNE_PERPLEXITY,
    UMAP_MIN_DIST,
    UMAP_NEIGHBORS,
)
from data_loader import load_20newsgroups_subset, load_chinese_ai_corpus, save_document_list
from preprocess import bm25_scores, build_count, build_tfidf, tokenize_docs
from recommender import build_knn_index, query_by_keywords, recommend
from topic_models import (
    assign_dominant_topic,
    numpy_svd_demo,
    top_terms_per_topic,
    train_lda,
    train_lsi,
)
from visualize import (
    plot_embedding_scatter,
    plot_singular_values,
    plot_topic_terms,
    reduce_2d,
)


def run_pipeline(name, docs, meta, lang="zh"):
    print(f"\n===== {name} =====")
    print(f"文档数: {len(docs)}")

    tokenized = tokenize_docs(docs, lang=lang)
    tfidf, tfidf_vec = build_tfidf(tokenized)
    count, count_vec = build_count(tokenized)
    bm25 = bm25_scores(tfidf)

    dense = tfidf.toarray()
    svd_info = numpy_svd_demo(dense, k=LSA_COMPONENTS)
    print(f"SVD 前 {LSA_COMPONENTS} 维能量占比: {svd_info['top_k_energy_ratio']:.2%}")

    lsi_model, lsi_topics = train_lsi(tfidf, n_components=LSA_COMPONENTS, random_state=RANDOM_STATE)
    lda_model, lda_topics = train_lda(count, n_topics=N_TOPICS, max_iter=LDA_MAX_ITER, random_state=RANDOM_STATE)

    labels = [m["label"] for m in meta]
    lsi_dom = assign_dominant_topic(lsi_topics)
    lda_dom = assign_dominant_topic(lda_topics)

    prefix = os.path.join(OUTPUT_DIR, name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_singular_values(svd_info["singular_values"], f"{prefix}_svd_spectrum.png")
    plot_topic_terms(
        top_terms_per_topic(lsi_model, tfidf_vec.get_feature_names_out(), kind="lsi"),
        f"{prefix}_lsi_topics.png",
        n_topics=LSA_COMPONENTS,
    )
    plot_topic_terms(
        top_terms_per_topic(lda_model, count_vec.get_feature_names_out(), kind="lda"),
        f"{prefix}_lda_topics.png",
        n_topics=N_TOPICS,
    )

    embeddings = lsi_topics
    for method in ("pca", "tsne", "umap"):
        try:
            coords = reduce_2d(
                embeddings,
                method=method,
                random_state=RANDOM_STATE,
                umap_neighbors=UMAP_NEIGHBORS,
                umap_min_dist=UMAP_MIN_DIST,
            )
            plot_embedding_scatter(
                coords,
                [f"T{lsi_dom[i]}" for i in range(len(docs))],
                f"{name} LSI 文档嵌入 — {method.upper()}",
                f"{prefix}_embed_{method}.png",
                annotate=(len(docs) <= 20),
            )
        except Exception as exc:
            print(f"[warn] {method} 可视化跳过: {exc}")

    nn = build_knn_index(embeddings, metric="cosine", n_neighbors=KNN_K + 1)
    query_idx = 0
    recs = recommend(nn, embeddings, query_idx, k=KNN_K)
    kw_hits = query_by_keywords(tfidf, tfidf_vec, ["人工智能", "模型"] if lang == "zh" else ["space", "nasa"])

    report = {
        "dataset": name,
        "n_docs": len(docs),
        "svd_energy_ratio": svd_info["top_k_energy_ratio"],
        "lsi_topics": top_terms_per_topic(lsi_model, tfidf_vec.get_feature_names_out(), kind="lsi"),
        "lda_topics": top_terms_per_topic(lda_model, count_vec.get_feature_names_out(), kind="lda"),
        "query_doc_id": query_idx,
        "query_preview": docs[query_idx][:120],
        "knn_recommendations": [
            {"doc_id": r["doc_id"], "distance": r["distance"], "preview": docs[r["doc_id"]][:120]}
            for r in recs
        ],
        "keyword_hits": [
            {"doc_id": h["doc_id"], "score": h["score"], "preview": docs[h["doc_id"]][:120]}
            for h in kw_hits
        ],
    }

    with open(f"{prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(f"{prefix}_recommend_demo.txt", "w", encoding="utf-8") as f:
        f.write(f"查询文档 #{query_idx}\n{docs[query_idx]}\n\nKNN 推荐:\n")
        for r in recs:
            f.write(f"  - #{r['doc_id']} (距离={r['distance']:.4f})\n    {docs[r['doc_id']][:100]}\n")

    print(f"推荐示例: 文档#{query_idx} → {[r['doc_id'] for r in recs]}")
    return report


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    zh_docs, zh_meta = load_chinese_ai_corpus()
    save_document_list(zh_docs, os.path.join(OUTPUT_DIR, "chinese_docs.txt"))
    zh_report = run_pipeline("chinese_ai", zh_docs, zh_meta, lang="zh")

    en_docs, en_meta = load_20newsgroups_subset()
    save_document_list(en_docs, os.path.join(OUTPUT_DIR, "newsgroups_docs.txt"))
    en_report = run_pipeline("newsgroups", en_docs, en_meta, lang="en")

    summary = {
        "chinese_ai": {"n_docs": zh_report["n_docs"], "svd_energy": zh_report["svd_energy_ratio"]},
        "newsgroups": {"n_docs": en_report["n_docs"], "svd_energy": en_report["svd_energy_ratio"]},
        "outputs": sorted(os.listdir(OUTPUT_DIR)),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n全部完成，输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
