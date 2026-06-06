# -*- coding: utf-8 -*-
"""作业三：路径与超参数。"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPO_ROOT = os.path.dirname(BASE_DIR)

HW1_DIR = os.path.join(REPO_ROOT, "homework_1")
HW2_DIR = os.path.join(REPO_ROOT, "homework_2")

AI_CORPUS = os.path.join(HW1_DIR, "ai_corpus.txt")
HW1_RAW = os.path.join(HW1_DIR, "output", "ai_raw_data.txt")
HW2_MIXED = os.path.join(HW2_DIR, "output", "mixed_raw_corpus.txt")
STOPWORDS = os.path.join(HW1_DIR, "hit_stopwords.txt")
USER_DICT_AI = os.path.join(HW2_DIR, "output", "user_dict_ai.txt")

# 主题模型
N_TOPICS = 5
LSA_COMPONENTS = 5
LDA_MAX_ITER = 30
RANDOM_STATE = 42

# 可视化
UMAP_NEIGHBORS = 8
UMAP_MIN_DIST = 0.25
TSNE_PERPLEXITY = 12

# 推荐
KNN_K = 5
