# -*- coding: utf-8 -*-
"""基于 jieba + 用户词典的分词。"""
import os

import jieba

from text_process import normalize_chars, split_documents


def jieba_lcut(text):
    """兼容部分精简/旧版发行版无 lcut 的情况：lcut 等价于 list(cut())。"""
    if hasattr(jieba, "lcut"):
        return jieba.lcut(text)
    return list(jieba.cut(text))


def load_stopwords(path):
    sw = set()
    if not os.path.exists(path):
        return sw
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                sw.add(w)
    return sw


def reset_jieba_user_dict(user_dict_path):
    """jieba 默认已加载内建词典；用户词典增量加载。"""
    jieba.initialize()
    if user_dict_path and os.path.exists(user_dict_path):
        jieba.load_userdict(user_dict_path)


def segment_document(text, stopwords):
    """单段文本 -> 词列表（去停用词、长度>=2）。"""
    t = normalize_chars(text)
    if not t:
        return []
    words = jieba_lcut(t)
    out = []
    for w in words:
        w = w.strip()
        if len(w) < 2 or w in stopwords:
            continue
        out.append(w)
    return out


def segment_corpus(raw_text, stopwords, user_dict_path):
    reset_jieba_user_dict(user_dict_path)

    docs_text = split_documents(raw_text)
    tokenized = []
    for d in docs_text:
        tokenized.append(segment_document(d, stopwords))
    return tokenized, docs_text
