# -*- coding: utf-8 -*-
"""语料采集：百度百科、百度资讯；可与作业一语料互补。"""
import os
import re

import requests
from bs4 import BeautifulSoup

from config import BASE_DIR, HEADERS


def collect_from_baidu_baike(keyword):
    url = "https://baike.baidu.com/item/" + keyword
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all(
            ["p", "div"], class_=re.compile(r"para|content")
        )
        text = "\n".join(
            p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
        )
        print("[百度百科] %s: %s (%d 字)" % (keyword, "ok" if text else "empty", len(text)))
        return text
    except Exception as e:
        print("[百度百科] %s: 失败 %s" % (keyword, e))
        return ""


def collect_from_baidu_news(keyword, page_count=2):
    all_text = []
    for page in range(page_count):
        url = "https://www.baidu.com/s?wd=%s&tn=news&pn=%d" % (keyword, page * 10)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
            for item in soup.find_all(["h3", "div", "span"]):
                txt = item.get_text(strip=True)
                if len(txt) > 15 and keyword[: min(2, len(keyword))] in txt:
                    all_text.append(txt)
        except Exception as e:
            print("[百度资讯] %s 第%d页: %s" % (keyword, page + 1, e))
    result = "\n".join(all_text)
    print("[百度资讯] %s: %d 字" % (keyword, len(result)))
    return result


def load_fallback_ai_corpus():
    """作业一目录下的备用语料（若存在）。"""
    p = os.path.join(BASE_DIR, "..", "homework_1", "ai_corpus.txt")
    p = os.path.normpath(p)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            t = f.read()
        print("[备用语料] homework_1/ai_corpus.txt: %d 字" % len(t))
        return t
    return ""


def build_mixed_corpus():
    """混合语料：AI + 财经，用于词典迭代与同句对比分词。"""
    chunks = []

    ai_keywords = ["人工智能", "深度学习", "机器学习", "自然语言处理", "大语言模型"]
    for kw in ai_keywords:
        t = collect_from_baidu_baike(kw)
        if t:
            chunks.append("【百科·AI】" + kw + "\n" + t)

    fin_keywords = ["股票", "上市公司", "证券投资基金", "央行", "货币政策"]
    for kw in fin_keywords:
        t = collect_from_baidu_baike(kw)
        if t:
            chunks.append("【百科·财经】" + kw + "\n" + t)

    n1 = collect_from_baidu_news("人工智能产业发展", page_count=1)
    if n1:
        chunks.append("【资讯·AI】\n" + n1)
    n2 = collect_from_baidu_news("A股市场行情", page_count=1)
    if n2:
        chunks.append("【资讯·财经】\n" + n2)

    chunks.append(load_fallback_ai_corpus())

    full = "\n\n".join(c for c in chunks if c)
    if len(full) < 500:
        print("[警告] 在线语料过少，使用内置短梗概补齐。")
        full = full + "\n\n" + _tiny_static_corpus()
    return full


def _tiny_static_corpus():
    return (
        "大语言模型通过预训练与微调，在多项自然语言处理任务上取得突破。"
        "央行通过公开市场操作调节流动性，影响债券市场与货币市场利率。"
        "上市公司披露年度报告，投资者关注净资产收益率与每股收益。"
        "深度学习利用反向传播训练神经网络，算力与数据规模共同驱动进步。"
    )


def save_raw(text, name="mixed_raw_corpus.txt"):
    from config import OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fp = os.path.join(OUTPUT_DIR, name)
    with open(fp, "w", encoding="utf-8") as f:
        f.write(text)
    print("[保存] 原始语料 -> %s" % fp)
    return fp
