import requests
from bs4 import BeautifulSoup
import jieba
import jieba.analyse
import re
import os
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "KaiTi"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_stopwords():
    sw = set()
    hit_path = os.path.join(BASE_DIR, "hit_stopwords.txt")
    if os.path.exists(hit_path):
        with open(hit_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    sw.add(w)
        print("[stopwords] 加载哈工大停用词表: %d 个" % len(sw))
    return sw

STOPWORDS = load_stopwords()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def collect_from_baidu_baike(keyword):
    """从百度百科爬取指定关键词的页面正文"""
    url = "https://baike.baidu.com/item/" + keyword
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all(
            ["p", "div"], class_=re.compile(r"para|content")
        )
        text = "\n".join(
            p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
        )
        status = "ok" if text else "empty"
        print("[百度百科] %s: %s (%d 字)" % (keyword, status, len(text)))
        return text
    except Exception as e:
        print("[百度百科] %s: 失败 %s" % (keyword, e))
        return ""


def collect_from_baidu_news(keyword, page_count=2):
    """从百度资讯搜索爬取AI相关新闻标题和摘要"""
    all_text = []
    for page in range(page_count):
        url = "https://www.baidu.com/s?wd=%s&tn=news&pn=%d" % (keyword, page * 10)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
            for item in soup.find_all(["h3", "div", "span"]):
                txt = item.get_text(strip=True)
                if len(txt) > 15 and keyword[0] in txt:
                    all_text.append(txt)
        except Exception as e:
            print("[百度资讯] 第%d页失败: %s" % (page + 1, e))
    result = "\n".join(all_text)
    print("[百度资讯] %s: %d 字" % (keyword, len(result)))
    return result


def get_backup_ai_corpus():
    """备用AI语料库"""
    backup_path = os.path.join(BASE_DIR, "ai_corpus.txt")
    if os.path.exists(backup_path):
        with open(backup_path, "r", encoding="utf-8") as f:
            corpus = f.read()
    else:
        corpus = ""
    print("[备用语料] 加载: %d 字" % len(corpus))
    return corpus


def save_raw_data(text, filename="ai_raw_data.txt"):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print("[保存] 原始数据 -> %s" % filepath)
    return filepath


def preprocess_text(text):
    """文本预处理：清洗 + 分词 + 去停用词"""
    text = re.sub(r"[a-zA-Z0-9]", "", text)
    text = re.sub(r"[^\u4e00-\u9fff]", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    words = jieba.lcut(text)
    words = [
        w.strip()
        for w in words
        if len(w.strip()) >= 2 and w.strip() not in STOPWORDS
    ]
    print("[预处理] 分词完成，共得到 %d 个有效词语" % len(words))
    return words


def get_word_frequency(words, top_n=30):
    counter = Counter(words)
    top_words = counter.most_common(top_n)
    df = pd.DataFrame(top_words, columns=["词语", "频次"])
    df.index = range(1, len(df) + 1)
    df.index.name = "排名"
    return df, counter


def extract_keywords(text, top_n=15):
    """使用TF-IDF提取关键词"""
    keywords = jieba.analyse.extract_tags(
        text, topK=top_n, withWeight=True,
        allowPOS=("n", "nr", "ns", "nt", "nz", "vn", "an"),
    )
    df = pd.DataFrame(keywords, columns=["关键词", "TF-IDF权重"])
    df.index = range(1, len(df) + 1)
    df.index.name = "排名"
    return df


def generate_wordcloud(word_freq, filename="ai_wordcloud.png"):
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    font_path = None
    for fp in font_candidates:
        if os.path.exists(fp):
            font_path = fp
            break
    wc = WordCloud(
        font_path=font_path, width=1200, height=800,
        background_color="white", max_words=100, max_font_size=120,
        min_font_size=12, colormap="viridis", collocations=False, margin=10,
    )
    wc.generate_from_frequencies(word_freq)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title("AI主题文本数据词云图", fontsize=20, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print("[可视化] 词云图 -> %s" % filepath)


def plot_top_words_bar(df, filename="ai_top_words_bar.png"):
    top20 = df.head(20)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Blues([(i + 5) / 25 for i in range(len(top20))])[::-1]
    bars = ax.barh(range(len(top20)), top20["频次"].values[::-1], color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["词语"].values[::-1], fontsize=13)
    ax.set_xlabel("出现频次", fontsize=13)
    ax.set_title("AI主题文本 — Top 20 高频词统计", fontsize=16, fontweight="bold")
    for bar, val in zip(bars, top20["频次"].values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print("[可视化] 柱状图 -> %s" % filepath)


def plot_keyword_weights(kw_df, filename="ai_keyword_weights.png"):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Oranges([(i + 4) / 19 for i in range(len(kw_df))])[::-1]
    bars = ax.barh(range(len(kw_df)), kw_df["TF-IDF权重"].values[::-1], color=colors)
    ax.set_yticks(range(len(kw_df)))
    ax.set_yticklabels(kw_df["关键词"].values[::-1], fontsize=13)
    ax.set_xlabel("TF-IDF 权重", fontsize=13)
    ax.set_title("AI主题文本 — TF-IDF 关键词权重", fontsize=16, fontweight="bold")
    for bar, val in zip(bars, kw_df["TF-IDF权重"].values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                "%.4f" % val, va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print("[可视化] TF-IDF图 -> %s" % filepath)


def plot_word_category_pie(words, filename="ai_category_pie.png"):
    counter = Counter(words)
    tech_kw = {"人工智能","深度学习","机器学习","神经网络","算法","模型","自然语言","计算机视觉","强化学习","知识图谱","多模态","智能","技术","数据","算力","芯片","大模型","网络","语言","训练","推理","计算","机器","学习"}
    app_kw = {"医疗","金融","教育","交通","制造","机器人","应用","领域","产业","场景","服务"}
    corp_kw = {"百度","阿里巴巴","腾讯","华为","科大讯飞","企业","高校","科研","研究院","研究"}
    trend_kw = {"发展","创新","突破","变革","未来","趋势","前沿","进展","升级","转型","演进"}
    ethic_kw = {"隐私","伦理","安全","风险","规范","法律","标准","治理","监管","政策"}
    cat_map = {"技术概念": tech_kw, "应用领域": app_kw, "企业机构": corp_kw, "发展趋势": trend_kw, "伦理治理": ethic_kw}
    counts = {}
    for cat, kws in cat_map.items():
        total = sum(counter.get(kw, 0) for kw in kws)
        if total > 0:
            counts[cat] = total
    if not counts:
        return
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(9, 9))
    labels = list(counts.keys())
    sizes = list(counts.values())
    explode = [0.05] * len(labels)
    colors_list = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, autopct="%1.1f%%",
        colors=colors_list[:len(labels)], startangle=140,
        textprops={"fontsize": 13},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    ax.set_title("AI主题词语 — 语义类别分布", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print("[可视化] 饼图 -> %s" % filepath)


def main():

    print("\n第一步：数据收集")
    all_texts = []
    for kw in ["人工智能", "深度学习", "大语言模型", "ChatGPT", "自然语言处理"]:
        text = collect_from_baidu_baike(kw)
        if text:
            all_texts.append(text)
    news = collect_from_baidu_news("人工智能技术发展")
    if news:
        all_texts.append(news)
    all_texts.append(get_backup_ai_corpus())
    full_text = "\n".join(all_texts)
    print("\n[汇总] 总共收集文本 %d 字" % len(full_text))
    save_raw_data(full_text)

    print("\n第二步：数据预处理（分词、去停用词）")
    words = preprocess_text(full_text)
    freq_df, word_counter = get_word_frequency(words, top_n=30)
    print("\n--- Top 30 高频词 ---")
    print(freq_df.to_string())
    kw_df = extract_keywords(full_text, top_n=15)
    print("\n--- TF-IDF Top 15 关键词 ---")
    print(kw_df.to_string())
    freq_df.to_csv(os.path.join(OUTPUT_DIR, "ai_word_frequency.csv"), encoding="utf-8-sig")
    kw_df.to_csv(os.path.join(OUTPUT_DIR, "ai_keywords_tfidf.csv"), encoding="utf-8-sig")

    print("\n第三步：数据可视化")
    generate_wordcloud(dict(word_counter))
    plot_top_words_bar(freq_df)
    plot_keyword_weights(kw_df)
    plot_word_category_pie(words)

    print("\n数据收集与预处理完成")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print("  - %s  (%s bytes)" % (f, "{:,}".format(size)))


if __name__ == "__main__":
    main()
