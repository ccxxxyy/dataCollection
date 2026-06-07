# -*- coding: utf-8 -*-
"""课程设计专属可视化：语料统计、新词评估、全流程示意图。"""
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_hw1_word_freq(word_freq, out_path, topn=12):
    words = [r["word"] for r in word_freq[:topn]]
    freqs = [r["freq"] for r in word_freq[:topn]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words[::-1], freqs[::-1], color="#55A868")
    ax.set_title("作业一语料高频词（人工智能主题）")
    ax.set_xlabel("词频")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hw1_tfidf(tfidf_rows, out_path, topn=12):
    words = [r["word"] for r in tfidf_rows[:topn]]
    weights = [r["weight"] for r in tfidf_rows[:topn]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words[::-1], weights[::-1], color="#C44E52")
    ax.set_title("作业一 TF-IDF 关键词权重")
    ax.set_xlabel("TF-IDF 权重")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_newword_burst(scored_words, out_path, topn=15):
    items = scored_words[:topn]
    if not items:
        return
    words = [x["word"] for x in items]
    scores = [x["burst_index"] for x in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(words[::-1], scores[::-1], color="#8172B2")
    ax.set_title("新词爆发力指数 Top（课程设计自定义评估）")
    ax.set_xlabel("爆发力指数 = 0.4·TF + 0.3·DF + 0.3·新颖度")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pipeline_overview(out_path):
    """绘制作业一至三联动流程示意图。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 2.0, "作业一\n语料采集\n词频/TF-IDF"),
        (2.5, 2.0, "作业二\n定制词典\nW2V+Annoy新词"),
        (4.7, 2.0, "作业三\nLSI/LDA降维\nKNN推荐"),
        (6.9, 2.0, "课程设计\n新词爆发力\n综合可视化"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y - 0.55), 1.8, 1.1, fill=True, color="#E8F4FD", ec="#4C72B0", lw=1.5))
        ax.text(x + 0.9, y, text, ha="center", va="center", fontsize=9)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.8
        x2 = boxes[i + 1][0]
        y = boxes[i][1]
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.set_title("课程设计：作业一至三数据联动全流程", fontsize=12, pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dict_growth_proxy(seed_n, final_n, out_path):
    """词典规模增长示意（基于作业二统计）。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["种子词", "扩展后"]
    vals = [seed_n, final_n]
    ax.bar(labels, vals, color=["#DD8452", "#4C72B0"])
    ax.set_title("作业二 AI 领域词典规模变化")
    ax.set_ylabel("词数")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, str(v), ha="center")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
