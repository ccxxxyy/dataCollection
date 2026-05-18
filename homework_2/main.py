# -*- coding: utf-8 -*-
"""
作业二入口：定制化词典流水线（结巴 + Word2Vec + Annoy + TF/DF 新词评估）。
运行（仓库根目录）： python homework_2/main.py
"""
import os

from config import (
    OUTPUT_DIR,
    SEED_AI,
    SEED_FINANCE,
    STOPWORDS_PATH,
)
from corpus_collector import build_mixed_corpus, save_raw
from dict_manager import load_word_set
from pipeline import compare_demo_sentences, run_pipeline
from visualize import plot_dict_growth, plot_token_freq_segmented


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== 作业2：采集 / 混合语料 ===\n")
    raw = build_mixed_corpus()
    save_raw(raw, "mixed_raw_corpus.txt")

    user_ai = os.path.join(OUTPUT_DIR, "user_dict_ai.txt")
    user_fin = os.path.join(OUTPUT_DIR, "user_dict_finance.txt")

    print("\n=== AI 领域：迭代扩展词典 ===\n")
    r_ai = run_pipeline(raw, STOPWORDS_PATH, SEED_AI, user_ai, "AI")

    print("\n=== 财经领域：迭代扩展词典 ===\n")
    r_fin = run_pipeline(raw, STOPWORDS_PATH, SEED_FINANCE, user_fin, "财经")

    for r in (r_ai, r_fin):
        print("\n----- %s 日志 -----" % r["name"])
        print("\n".join(r["log"]))

    demo_sents = [
        "上市公司披露年报后投资者关注净资产收益率与市盈率变化。",
        "大语言模型通过预训练和微调提升自然语言处理任务的表现。",
        "央行通过逆回购调节银行间市场流动性并影响债券收益率曲线。",
    ]
    rows = compare_demo_sentences(
        demo_sents,
        STOPWORDS_PATH,
        [user_ai, user_fin],
        ["AI词典分词", "财经词典分词"],
    )
    demo_path = os.path.join(OUTPUT_DIR, "segment_compare_demo.txt")
    with open(demo_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["句子"] + "\n")
            f.write("  AI:   " + row["AI词典分词"] + "\n")
            f.write("  财经: " + row["财经词典分词"] + "\n\n")
    print("\n[保存] 分词对比 -> %s" % demo_path)

    # 若两轮历史长度不一致，对齐到最短长度绘图
    h_ai = r_ai["history_sizes"]
    h_fin = r_fin["history_sizes"]
    n = min(len(h_ai), len(h_fin))
    plot_dict_growth(h_ai[:n], h_fin[:n])
    plot_token_freq_segmented(r_ai["final_tokenized"], fname="token_freq_ai_dict.png")
    plot_token_freq_segmented(
        r_fin["final_tokenized"], fname="token_freq_finance_dict.png"
    )

    stat = os.path.join(OUTPUT_DIR, "dictionary_stats.txt")
    with open(stat, "w", encoding="utf-8") as f:
        f.write("AI 种子词数: %d\n" % len(load_word_set(SEED_AI)))
        f.write("财经 种子词数: %d\n" % len(load_word_set(SEED_FINANCE)))
        f.write("AI 最终词典词数: %d\n" % len(r_ai["domain"]))
        f.write("财经 最终词典词数: %d\n" % len(r_fin["domain"]))
    print("[保存] %s" % stat)

    print("\n完成。输出目录: %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()
