# -*- coding: utf-8 -*-
"""Matplotlib 可视化。"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "KaiTi"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_dict_growth(hist_ai, hist_fin, fname="dict_growth.png"):
    fp = os.path.join(OUTPUT_DIR, fname)
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = range(1, len(hist_ai) + 1)
    ax.plot(xs, hist_ai, "o-", label="AI 定制词典", linewidth=2)
    ax.plot(xs, hist_fin, "s--", label="财经 定制词典", linewidth=2)
    ax.set_xlabel("迭代轮次（从 1 开始）")
    ax.set_ylabel("词典词数（规模）")
    ax.set_title("定制化词典迭代扩展 — 规模变化")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close()
    print("[图] %s" % fp)


def plot_token_freq_segmented(tokenized_docs, top_n=20, fname="token_freq_after_ai_dict.png"):
    """分词后词频 Top N（用于展示分词效果）。"""
    from collections import Counter

    c = Counter()
    for d in tokenized_docs:
        c.update(d)
    top = c.most_common(top_n)
    if not top:
        return
    words, vals = zip(*top)
    fp = os.path.join(OUTPUT_DIR, fname)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(words)), list(vals)[::-1], color="#4C72B0")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(list(words)[::-1])
    ax.set_xlabel("频次")
    ax.set_title("基于定制词典分词后的 Top %d 词频" % top_n)
    plt.tight_layout()
    plt.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close()
    print("[图] %s" % fp)
