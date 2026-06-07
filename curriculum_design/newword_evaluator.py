# -*- coding: utf-8 -*-
"""
新词评估机制（课程设计亮点）：
结合作业二 TF/DF 过滤思想，设计「新词爆发力指数」=
  0.4 * TF_norm + 0.3 * DF_norm + 0.3 * 语义新颖度（相对种子词的平均编辑距离）
"""
import numpy as np

from settings import NEWWORD_TOP_N
from hw3_imports import build_tfidf, tokenize_docs


def _edit_distance(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev, dp[j] = dp[j], cur
    return dp[lb]


def _novelty_score(word, seeds):
    if not seeds:
        return 1.0
    dists = [_edit_distance(word, s) for s in seeds]
    return min(1.0, np.mean(dists) / max(len(word), 3))


def evaluate_new_words(docs, seed_words, user_dict_words, topn=NEWWORD_TOP_N):
    """对作业二扩展词典中的新词打分排序。"""
    seeds = set(seed_words)
    new_words = [w for w in user_dict_words if w not in seeds and len(w) >= 2]
    if not new_words:
        return []

    tokenized = tokenize_docs(docs, lang="zh")
    tfidf, vec = build_tfidf(tokenized)
    vocab = set(vec.get_feature_names_out())

    doc_freq = {}
    term_freq = {}
    for toks in tokenized:
        seen = set()
        for t in toks:
            term_freq[t] = term_freq.get(t, 0) + 1
            if t not in seen:
                doc_freq[t] = doc_freq.get(t, 0) + 1
                seen.add(t)

    scored = []
    for w in new_words:
        if w not in vocab and w not in term_freq:
            continue
        tf = term_freq.get(w, 0)
        df = doc_freq.get(w, 0)
        if tf == 0:
            continue
        novelty = _novelty_score(w, list(seeds))
        scored.append({
            "word": w,
            "tf": tf,
            "df": df,
            "novelty": round(novelty, 4),
            "burst_index": 0.0,
        })

    if not scored:
        return []

    max_tf = max(s["tf"] for s in scored) or 1
    max_df = max(s["df"] for s in scored) or 1
    for s in scored:
        tf_n = s["tf"] / max_tf
        df_n = s["df"] / max_df
        s["burst_index"] = round(0.4 * tf_n + 0.3 * df_n + 0.3 * s["novelty"], 4)

    scored.sort(key=lambda x: (-x["burst_index"], -x["tf"], -x["df"]))
    return scored[:topn]
