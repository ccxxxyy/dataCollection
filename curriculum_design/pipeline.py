# -*- coding: utf-8 -*-
"""课程设计主流程：整合作业一至三，输出报告与图表。"""
import json
import os
import shutil

from settings import (
    HW2_DICT_GROWTH,
    KNN_K,
    LDA_MAX_ITER,
    LSA_COMPONENTS,
    N_TOPICS,
    OUTPUT_DIR,
    RANDOM_STATE,
    UMAP_MIN_DIST,
    UMAP_NEIGHBORS,
)
from data_integration import load_integrated_corpus
from hw3_imports import (
    assign_dominant_topic,
    bm25_scores,
    build_count,
    build_knn_index,
    build_tfidf,
    numpy_svd_demo,
    plot_embedding_scatter,
    plot_singular_values,
    plot_topic_terms,
    query_by_keywords,
    recommend,
    reduce_2d,
    tokenize_docs,
    top_terms_per_topic,
    train_lda,
    train_lsi,
)
from newword_evaluator import evaluate_new_words
from visualize_extra import (
    plot_dict_growth_proxy,
    plot_hw1_tfidf,
    plot_hw1_word_freq,
    plot_newword_burst,
    plot_pipeline_overview,
)


def _run_topic_pipeline(name, docs, meta, lang="zh"):
    tokenized = tokenize_docs(docs, lang=lang)
    tfidf, tfidf_vec = build_tfidf(tokenized)
    count, count_vec = build_count(tokenized)
    bm25_scores(tfidf)

    dense = tfidf.toarray()
    svd_info = numpy_svd_demo(dense, k=LSA_COMPONENTS)

    lsi_model, lsi_topics = train_lsi(tfidf, n_components=LSA_COMPONENTS, random_state=RANDOM_STATE)
    lda_model, lda_topics = train_lda(count, n_topics=N_TOPICS, max_iter=LDA_MAX_ITER, random_state=RANDOM_STATE)
    lsi_dom = assign_dominant_topic(lsi_topics)

    prefix = os.path.join(OUTPUT_DIR, name)
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
    kw_hits = query_by_keywords(
        tfidf, tfidf_vec,
        ["人工智能", "模型"] if lang == "zh" else ["space", "nasa"],
    )

    return {
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


def run_full_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    integrated = load_integrated_corpus()

    # 课程设计专属图表
    plot_pipeline_overview(os.path.join(OUTPUT_DIR, "pipeline_overview.png"))
    plot_hw1_word_freq(integrated["hw1_word_freq"], os.path.join(OUTPUT_DIR, "hw1_word_freq.png"))
    plot_hw1_tfidf(integrated["hw1_tfidf"], os.path.join(OUTPUT_DIR, "hw1_tfidf.png"))

    stats = integrated["hw2_dict_stats"]
    seed_n = int(stats.get("AI 种子词数", "31"))
    final_n = int(stats.get("AI 最终词典词数", "223"))
    plot_dict_growth_proxy(seed_n, final_n, os.path.join(OUTPUT_DIR, "hw2_dict_growth.png"))
    if os.path.isfile(HW2_DICT_GROWTH):
        shutil.copy2(HW2_DICT_GROWTH, os.path.join(OUTPUT_DIR, "hw2_dict_growth_curve.png"))

    zh_docs = integrated["chinese_ai"]["docs"]
    newword_scores = evaluate_new_words(
        zh_docs,
        integrated["seed_words"],
        integrated["user_dict_words"],
    )
    plot_newword_burst(newword_scores, os.path.join(OUTPUT_DIR, "newword_burst_index.png"))

    zh_report = _run_topic_pipeline("chinese_ai", zh_docs, integrated["chinese_ai"]["meta"], lang="zh")
    en_docs = integrated["newsgroups"]["docs"]
    en_report = _run_topic_pipeline("newsgroups", en_docs, integrated["newsgroups"]["meta"], lang="en")

    report = {
        "title": "基于作业一至三数据的人工智能文本挖掘综合课程设计",
        "integrated": {
            "n_chinese_docs": len(zh_docs),
            "n_english_docs": len(en_docs),
            "hw1_corpus_lines": integrated["chinese_ai"]["n_hw1_lines"],
            "hw2_dict_stats": stats,
            "seed_word_count": len(integrated["seed_words"]),
            "user_dict_count": len(integrated["user_dict_words"]),
        },
        "hw1_top_words": integrated["hw1_word_freq"],
        "hw1_tfidf": integrated["hw1_tfidf"],
        "newword_scores": newword_scores,
        "chinese_ai": zh_report,
        "newsgroups": en_report,
    }

    with open(os.path.join(OUTPUT_DIR, "curriculum_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = {
        "chinese_ai": {"n_docs": zh_report["n_docs"], "svd_energy": zh_report["svd_energy_ratio"]},
        "newsgroups": {"n_docs": en_report["n_docs"], "svd_energy": en_report["svd_energy_ratio"]},
        "newword_top5": [w["word"] for w in newword_scores[:5]],
        "outputs": sorted(os.listdir(OUTPUT_DIR)),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("课程设计流水线完成，输出目录:", OUTPUT_DIR)
    return report
