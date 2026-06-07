# -*- coding: utf-8 -*-
"""课程设计：路径与超参数。"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPO_ROOT = os.path.dirname(BASE_DIR)

HW1_DIR = os.path.join(REPO_ROOT, "homework_1")
HW2_DIR = os.path.join(REPO_ROOT, "homework_2")
HW3_DIR = os.path.join(REPO_ROOT, "homewprk_3")

HW1_CORPUS = os.path.join(HW1_DIR, "ai_corpus.txt")
HW1_RAW = os.path.join(HW1_DIR, "output", "ai_raw_data.txt")
HW1_OUTPUT = os.path.join(HW1_DIR, "output")
HW1_WORD_FREQ = os.path.join(HW1_OUTPUT, "ai_word_frequency.csv")
HW1_TFIDF = os.path.join(HW1_OUTPUT, "ai_keywords_tfidf.csv")

HW2_OUTPUT = os.path.join(HW2_DIR, "output")
HW2_MIXED = os.path.join(HW2_OUTPUT, "mixed_raw_corpus.txt")
HW2_USER_DICT_AI = os.path.join(HW2_OUTPUT, "user_dict_ai.txt")
HW2_DICT_STATS = os.path.join(HW2_OUTPUT, "dictionary_stats.txt")
HW2_DICT_GROWTH = os.path.join(HW2_OUTPUT, "dict_growth.png")

STOPWORDS = os.path.join(HW1_DIR, "hit_stopwords.txt")
SEED_AI = os.path.join(HW2_DIR, "seed_terms_ai.txt")

N_TOPICS = 5
LSA_COMPONENTS = 5
LDA_MAX_ITER = 30
RANDOM_STATE = 42
KNN_K = 5

UMAP_NEIGHBORS = 8
UMAP_MIN_DIST = 0.25

NEWWORD_TOP_N = 20
