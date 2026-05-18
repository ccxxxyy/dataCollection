# -*- coding: utf-8 -*-
"""文本切块、清洗、TF/DF 统计。"""
import re
from collections import Counter


def split_documents(raw_text):
    """以空行分段作为『文档』，用于 DF 统计。"""
    parts = re.split(r"\n\s*\n", raw_text.strip())
    docs = [p.strip() for p in parts if p.strip()]
    if not docs:
        docs = [raw_text.strip()]
    return docs


def normalize_chars(text):
    """保留中文，其它变空白，压缩连续空白。"""
    text = re.sub(r"[^\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_tf_df(tokenized_docs, candidate_words):
    """tokenized_docs: list[list[str]]；返回每个候选词的 (tf, df)。"""
    tf_counter = Counter()
    df_real = Counter()
    cand = set(candidate_words)
    for toks in tokenized_docs:
        tf_counter.update(toks)
        for w in set(toks):
            if w in cand:
                df_real[w] += 1
    return {w: (tf_counter.get(w, 0), df_real.get(w, 0)) for w in candidate_words}
