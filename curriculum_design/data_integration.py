# -*- coding: utf-8 -*-
"""整合作业一、二、三语料与统计结果。"""
import csv
import os
import re

from sklearn.datasets import fetch_20newsgroups

from settings import (
    HW1_CORPUS,
    HW1_RAW,
    HW1_TFIDF,
    HW1_WORD_FREQ,
    HW2_DICT_STATS,
    HW2_MIXED,
    HW2_USER_DICT_AI,
    SEED_AI,
)


def _read_lines(path, min_len=20, max_docs=None):
    if not os.path.isfile(path):
        return []
    docs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if len(text) >= min_len:
                docs.append(text)
            if max_docs and len(docs) >= max_docs:
                break
    return docs


def load_hw1_word_freq(topn=15):
    rows = []
    if not os.path.isfile(HW1_WORD_FREQ):
        return rows
    with open(HW1_WORD_FREQ, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"word": row["词语"], "freq": int(row["频次"])})
            if len(rows) >= topn:
                break
    return rows


def load_hw1_tfidf(topn=15):
    rows = []
    if not os.path.isfile(HW1_TFIDF):
        return rows
    with open(HW1_TFIDF, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"word": row["关键词"], "weight": float(row["TF-IDF权重"])})
            if len(rows) >= topn:
                break
    return rows


def load_hw2_dict_stats():
    stats = {}
    if not os.path.isfile(HW2_DICT_STATS):
        return stats
    with open(HW2_DICT_STATS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line or "：" in line:
                parts = re.split(r"[:：]", line, maxsplit=1)
                if len(parts) == 2:
                    stats[parts[0].strip()] = parts[1].strip()
    return stats


def load_seed_words():
    if not os.path.isfile(SEED_AI):
        return set()
    with open(SEED_AI, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_user_dict_words():
    words = []
    if not os.path.isfile(HW2_USER_DICT_AI):
        return words
    with open(HW2_USER_DICT_AI, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                words.append(parts[0])
    return words


def _split_paragraphs(raw, min_len=40, max_docs=80):
    parts = re.split(r"[\n\r]+", raw)
    docs = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= min_len:
            docs.append(p)
        if len(docs) >= max_docs:
            break
    return docs


def load_chinese_ai_corpus():
    docs = _read_lines(HW1_CORPUS, min_len=15)
    labels = ["人工智能综述"] * len(docs)
    if os.path.isfile(HW2_MIXED):
        with open(HW2_MIXED, "r", encoding="utf-8", errors="ignore") as f:
            extra = _split_paragraphs(f.read(), min_len=50, max_docs=40)
        docs.extend(extra)
        labels.extend(["混合语料"] * len(extra))
    if len(docs) < 12 and os.path.isfile(HW1_RAW):
        with open(HW1_RAW, "r", encoding="utf-8", errors="ignore") as f:
            extra = _split_paragraphs(f.read(), min_len=60, max_docs=30)
        docs.extend(extra)
        labels.extend(["采集文本"] * len(extra))
    meta = [{"source": "chinese_ai", "label": lb} for lb in labels]
    return docs, meta


def load_20newsgroups_subset():
    categories = ["sci.space", "comp.graphics", "rec.sport.baseball", "talk.politics.misc"]
    data = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True,
        random_state=42, remove=("headers", "footers", "quotes"),
    )
    docs = [t.strip() for t in data.data if len(t.strip()) > 80][:120]
    labels = [data.target_names[t] for t in data.target[: len(docs)]]
    meta = [{"source": "20newsgroups", "label": lb} for lb in labels]
    return docs, meta


def load_integrated_corpus():
    """作业一 + 作业二 + 作业三统一语料。"""
    zh_docs, zh_meta = load_chinese_ai_corpus()
    en_docs, en_meta = load_20newsgroups_subset()
    hw1_only = _read_lines(HW1_CORPUS, min_len=15)
    return {
        "chinese_ai": {"docs": zh_docs, "meta": zh_meta, "n_hw1_lines": len(hw1_only)},
        "newsgroups": {"docs": en_docs, "meta": en_meta},
        "hw2_mixed_exists": os.path.isfile(HW2_MIXED),
        "hw1_word_freq": load_hw1_word_freq(),
        "hw1_tfidf": load_hw1_tfidf(),
        "hw2_dict_stats": load_hw2_dict_stats(),
        "seed_words": sorted(load_seed_words()),
        "user_dict_words": load_user_dict_words(),
    }
