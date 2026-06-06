# -*- coding: utf-8 -*-
"""加载作业一、作业二语料，并准备 20 Newsgroups 英文子集。"""
import os
import re

from sklearn.datasets import fetch_20newsgroups

from config import AI_CORPUS, HW1_RAW, HW2_MIXED


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
    """以作业一 ai_corpus 为主，辅以作业二混合语料片段。"""
    docs = _read_lines(AI_CORPUS, min_len=15)
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
    categories = [
        "sci.space",
        "comp.graphics",
        "rec.sport.baseball",
        "talk.politics.misc",
    ]
    data = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )
    docs = [t.strip() for t in data.data if len(t.strip()) > 80][:120]
    labels = [data.target_names[t] for t in data.target[: len(docs)]]
    meta = [{"source": "20newsgroups", "label": lb} for lb in labels]
    return docs, meta


def save_document_list(docs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(f"--- doc {i} ---\n{d}\n\n")
