# -*- coding: utf-8 -*-
"""作业二：路径与超参数。"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STOPWORDS_PATH = os.path.join(BASE_DIR, "hit_stopwords.txt")
SEED_AI = os.path.join(BASE_DIR, "seed_terms_ai.txt")
SEED_FINANCE = os.path.join(BASE_DIR, "seed_terms_finance.txt")

# Word2Vec / Annoy / 新词挖掘
W2V_VECTOR_SIZE = 128
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 10
W2V_WORKERS = 4
ANNOY_NTREES = 20
K_NEIGHBORS = 12
MIN_TF = 3
MIN_DF = 2
MAX_ITERATIONS = 6
USER_DICT_FREQ = 10000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
