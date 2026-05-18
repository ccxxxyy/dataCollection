# -*- coding: utf-8 -*-
"""定制化词典文件的读写（jieba load_userdict 格式：词 词频）。"""
import os

def load_word_set(path):
    words = set()
    if not os.path.exists(path):
        return words
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            words.add(w)
    return words


def write_user_dict(word_set, path, default_freq=10000):
    """写入 jieba 用户词典：每行 `词 词频`。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    sorted_words = sorted(word_set)
    with open(path, "w", encoding="utf-8") as f:
        for w in sorted_words:
            f.write("%s %d\n" % (w, default_freq))
    print("[词典] 写入 %d 个词 -> %s" % (len(sorted_words), path))


def seed_dict(seed_path, out_user_dict_path, default_freq=10000):
    """从种子词表初始化用户词典文件。"""
    s = load_word_set(seed_path)
    write_user_dict(s, out_user_dict_path, default_freq)
    return set(s)
