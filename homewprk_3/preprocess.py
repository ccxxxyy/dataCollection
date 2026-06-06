# -*- coding: utf-8 -*-
"""中文分词与英文简单分词，构建 TF / TF-IDF / BM25 矩阵。"""
import os
import re

import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import STOPWORDS, USER_DICT_AI


def _jieba_lcut(text):
    """兼容 jieba 0.34（仅 cut）与新版（lcut）。"""
    if hasattr(jieba, "lcut"):
        return jieba.lcut(text)
    return list(jieba.cut(text))


def load_stopwords(path=STOPWORDS):
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return {line.strip() for line in f if line.strip()}


def _load_user_dict(path=USER_DICT_AI):
    if os.path.isfile(path):
        jieba.load_userdict(path)


def tokenize_chinese(text, stopwords):
    _load_user_dict()
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", " ", text)
    tokens = [
        w.strip()
        for w in _jieba_lcut(text)
        if len(w.strip()) >= 2 and w.strip() not in stopwords
    ]
    return tokens


def tokenize_english(text, stopwords):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [w for w in text.split() if len(w) >= 3 and w not in stopwords]
    return tokens


def tokenize_docs(docs, lang="zh"):
    stopwords = load_stopwords()
    if lang == "zh":
        return [tokenize_chinese(d, stopwords) for d in docs]
    eng_stop = {
        "the", "and", "for", "that", "with", "this", "from", "have", "are",
        "was", "were", "been", "will", "would", "could", "about", "into",
        "also", "they", "their", "there", "what", "when", "which", "than",
    }
    stopwords = stopwords | eng_stop
    return [tokenize_english(d, stopwords) for d in docs]


def tokens_to_texts(tokenized):
    return [" ".join(toks) for toks in tokenized]


def build_tfidf(tokenized, max_features=2000):
    texts = tokens_to_texts(tokenized)
    vec = TfidfVectorizer(max_features=max_features, min_df=1, max_df=0.95)
    matrix = vec.fit_transform(texts)
    return matrix, vec


def build_count(tokenized, max_features=2000):
    texts = tokens_to_texts(tokenized)
    vec = CountVectorizer(max_features=max_features, min_df=1, max_df=0.95)
    matrix = vec.fit_transform(texts)
    return matrix, vec


def bm25_scores(tfidf_matrix):
    """基于 TF-IDF 稀疏矩阵的简化 BM25 风格打分（用于展示）。"""
    X = tfidf_matrix.tocsr().astype(np.float64)
    doc_len = np.asarray(X.sum(axis=1)).ravel()
    avgdl = doc_len.mean() if len(doc_len) else 1.0
    k1, b = 1.2, 0.75
    idf = np.log(1 + tfidf_matrix.shape[0] / (1 + np.diff(tfidf_matrix.tocsc().indptr)))
    scores = X.copy()
    for i in range(scores.shape[0]):
        dl = doc_len[i] or 1.0
        row = scores.getrow(i)
        for j in row.indices:
            tf = row.data[list(row.indices).index(j)]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            scores[i, j] = idf[j] * (tf * (k1 + 1)) / (denom + 1e-9)
    return scores
