# -*- coding: utf-8 -*-
"""TF/DF 新词过滤与迭代流水线。"""
import os

from config import (
    ANNOY_NTREES,
    K_NEIGHBORS,
    MAX_ITERATIONS,
    MIN_DF,
    MIN_TF,
    OUTPUT_DIR,
    USER_DICT_FREQ,
    W2V_EPOCHS,
    W2V_MIN_COUNT,
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    W2V_WORKERS,
)
from dict_manager import load_word_set, write_user_dict
from segmentation import load_stopwords, segment_corpus
from text_process import compute_tf_df
from w2v_annoy_miner import knn_candidates, train_embedding_model


def _filter_candidates(candidates, domain, stopwords, tokenized_docs, min_tf, min_df):
    """按 TF、DF 与停用词、长度过滤。"""
    cand_list = []
    for w in candidates:
        if w in domain or len(w) < 2 or w in stopwords:
            continue
        cand_list.append(w)
    if not cand_list:
        return []
    stats = compute_tf_df(tokenized_docs, cand_list)
    accepted = []
    for w in cand_list:
        tf, df = stats[w]
        if tf >= min_tf and df >= min_df:
            accepted.append((w, tf, df))
    accepted.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return accepted


def run_pipeline(raw_text, stopwords_path, seed_path, user_dict_path, name):
    """单领域：迭代扩展用户词典并记录日志。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_lines = []
    domain = set(load_word_set(seed_path))
    write_user_dict(domain, user_dict_path, USER_DICT_FREQ)

    history_sizes = []
    all_new = []

    for it in range(1, MAX_ITERATIONS + 1):
        tokenized, _ = segment_corpus(raw_text, load_stopwords(stopwords_path), user_dict_path)
        history_sizes.append(len(domain))
        log_lines.append("== %s 迭代 %d | 词典规模 %d ==" % (name, it, len(domain)))

        embed_model, gensim_model = train_embedding_model(
            tokenized,
            W2V_VECTOR_SIZE,
            W2V_WINDOW,
            W2V_MIN_COUNT,
            W2V_EPOCHS,
            W2V_WORKERS,
        )
        cand = knn_candidates(
            embed_model,
            gensim_model,
            domain,
            K_NEIGHBORS,
            ANNOY_NTREES,
        )
        accepted = _filter_candidates(
            cand, domain, load_stopwords(stopwords_path), tokenized, MIN_TF, MIN_DF
        )
        if not accepted:
            log_lines.append("无满足 TF/DF 的新词，停止迭代。")
            break
        new_words = [a[0] for a in accepted[:32]]
        log_lines.append(
            "本批采纳新词（示例）: "
            + "、".join("%s(tf=%d,df=%d)" % (a[0], a[1], a[2]) for a in accepted[:15])
        )
        for w in new_words:
            domain.add(w)
            all_new.append(w)
        write_user_dict(domain, user_dict_path, USER_DICT_FREQ)

    return {
        "name": name,
        "domain": domain,
        "user_dict_path": user_dict_path,
        "history_sizes": history_sizes,
        "log": log_lines,
        "new_words": all_new,
        "final_tokenized": segment_corpus(
            raw_text, load_stopwords(stopwords_path), user_dict_path
        )[0],
    }


def compare_demo_sentences(sentences, stopwords_path, dict_paths, labels):
    """同一批例句，不同用户词典下的分词对比。"""
    stop = load_stopwords(stopwords_path)
    rows = []
    for s in sentences:
        row = {"句子": s}
        for path, lab in zip(dict_paths, labels):
            toks, _ = segment_corpus(s, stop, path)
            flat = []
            for t in toks:
                flat.extend(t)
            row[lab] = "/".join(flat)
        rows.append(row)
    return rows
