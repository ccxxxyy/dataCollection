# -*- coding: utf-8 -*-
"""安全导入 homewprk_3 模块，避免与 curriculum_design.config 冲突。"""
import importlib.util
import os
import sys

from settings import HW3_DIR


def _load(name, filename):
    path = os.path.join(HW3_DIR, filename)
    spec = importlib.util.spec_from_file_location(f"hw3_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"hw3_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_hw3_config = _load("config", "config.py")
sys.modules["config"] = _hw3_config
_hw3_preprocess = _load("preprocess", "preprocess.py")
_hw3_topic = _load("topic_models", "topic_models.py")
_hw3_recommender = _load("recommender", "recommender.py")
_hw3_visualize = _load("visualize", "visualize.py")

tokenize_docs = _hw3_preprocess.tokenize_docs
build_tfidf = _hw3_preprocess.build_tfidf
build_count = _hw3_preprocess.build_count
bm25_scores = _hw3_preprocess.bm25_scores

numpy_svd_demo = _hw3_topic.numpy_svd_demo
train_lsi = _hw3_topic.train_lsi
train_lda = _hw3_topic.train_lda
top_terms_per_topic = _hw3_topic.top_terms_per_topic
assign_dominant_topic = _hw3_topic.assign_dominant_topic

build_knn_index = _hw3_recommender.build_knn_index
recommend = _hw3_recommender.recommend
query_by_keywords = _hw3_recommender.query_by_keywords

plot_embedding_scatter = _hw3_visualize.plot_embedding_scatter
plot_singular_values = _hw3_visualize.plot_singular_values
plot_topic_terms = _hw3_visualize.plot_topic_terms
reduce_2d = _hw3_visualize.reduce_2d
