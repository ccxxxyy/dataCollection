# 作业一 数据收集与预处理

## 一、作业要求

1. 准备一份热身作业
   - a) 收集，整理一份个人（文本）数据集，内容、体裁、格式、创意不限
   - b) 参考暗恋日记
2. 将整理好的数据预处理，进行展示
   - a) 例如示例中的词云展示
3. 加分/得分/扣分项：
   - a) 有趣、好的创意、数据
   - b) 生动的表达方式，图片、图表（Matplotlib、Jupyter notebook等）
   - c) 正式的书写格式，参考毕业设计格式
   - d) 个人作业
   - e) 不允许抄袭、参考要列出出处，并作为文献参考

---

## 二、作业内容介绍、实现思路

### 2.1 选题背景

本次作业选择 **"人工智能（AI）"** 作为数据收集的主题。人工智能是当前最受关注的科技领域之一，从深度学习到大语言模型，从计算机视觉到自动驾驶，AI技术正在深刻改变人类社会的方方面面。选择该主题进行文本数据收集与分析，既具有现实意义，也能充分体现文本信息挖掘的核心流程。

### 2.2 数据来源

本次作业采用**多源数据收集**策略，从以下渠道获取AI相关文本数据：

| 数据来源 | 获取方式 | 说明 |
|---------|---------|------|
| 百度百科 | 网页爬虫（requests + BeautifulSoup） | 爬取"人工智能""深度学习""大语言模型""ChatGPT""自然语言处理"等词条 |
| 百度资讯 | 网页爬虫（requests + BeautifulSoup） | 搜索"人工智能技术发展"获取最新新闻标题和摘要 |
| 备用语料库 | 本地文件 | 精心编写的AI领域综述文本，确保数据充足 |

### 2.3 技术栈

- **Python 3** — 编程语言
- **requests** — HTTP请求库，用于网页数据获取
- **BeautifulSoup** — HTML解析库，提取网页正文
- **jieba** — 中文分词工具，支持TF-IDF关键词提取
- **wordcloud** — 词云生成库
- **matplotlib** — 数据可视化绑库
- **pandas** — 数据分析库

### 2.4 实现流程

整体流程遵循"数据收集 → 数据预处理 → 数据可视化"的流水线（pipeline）设计：

```
数据收集（爬虫/API/本地）
        ↓
    原始文本存储
        ↓
  文本预处理（清洗、分词、去停用词）
        ↓
  统计分析（词频统计、TF-IDF提取）
        ↓
  数据可视化（词云、柱状图、饼图）
```

**第一步：数据收集**

使用 `requests` 库发送HTTP请求到百度百科和百度资讯，通过 `BeautifulSoup` 解析HTML页面，提取正文段落文本。同时加载本地备用语料库作为补充。所有收集到的文本合并后保存为 `ai_raw_data.txt`。

**第二步：数据预处理**

1. **文本清洗**：使用正则表达式去除英文字符、数字和特殊符号，只保留中文字符
2. **中文分词**：使用 `jieba` 分词库对清洗后的文本进行分词
3. **去停用词**：构建停用词表，过滤"的""了""在"等无意义词汇
4. **主题过滤**：构建AI领域白名单词表，只保留与AI主题相关的词语，确保分析结果的纯净性
5. **TF-IDF关键词提取**：使用 `jieba.analyse` 的TF-IDF算法提取文本关键词及其权重

**第三步：数据可视化**

生成四种可视化图表：
- **词云图**：直观展示AI主题的核心词汇分布
- **Top 20高频词柱状图**：定量展示词频统计结果
- **TF-IDF关键词权重图**：展示关键词的重要性排序
- **语义类别饼图**：将词语按"技术概念""应用领域""企业机构""发展趋势""伦理治理"五大类别进行分类统计

---

## 三、实现代码部分

### 3.1 核心代码

完整代码见 `homework_1/ai_data_collection.py`，以下为核心部分展示。

**数据收集 — 百度百科爬虫：**

```python
def collect_from_baidu_baike(keyword):
    url = "https://baike.baidu.com/item/" + keyword
    headers = {"User-Agent": "Mozilla/5.0 ..."}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.find_all(["p", "div"], class_=re.compile(r"para|content"))
    text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return text
```

**数据预处理 — 分词与过滤：**

```python
def preprocess_text(text):
    text = re.sub(r"[a-zA-Z0-9]", "", text)          # 去除英文和数字
    text = re.sub(r"[^\u4e00-\u9fff]", " ", text)     # 只保留中文
    words = jieba.lcut(text)                           # jieba分词
    words = [w for w in words
             if len(w) >= 2
             and w not in STOPWORDS
             and w in AI_WHITELIST]                    # 去停用词 + 主题过滤
    return words
```

**数据可视化 — 词云生成：**

```python
def generate_wordcloud(word_freq):
    wc = WordCloud(
        font_path="C:/Windows/Fonts/msyh.ttc",
        width=1200, height=800,
        background_color="white", max_words=100,
        colormap="viridis", collocations=False,
    )
    wc.generate_from_frequencies(word_freq)
    plt.imshow(wc, interpolation="bilinear")
    plt.title("AI主题文本数据词云图")
    plt.axis("off")
    plt.savefig("homework_1/output/ai_wordcloud.png", dpi=200)
```

### 3.2 依赖安装

```bash
pip install requests beautifulsoup4 jieba wordcloud matplotlib pandas
```

### 3.3 运行方式

```bash
python homework_1/ai_data_collection.py
```

---

## 四、运行结果

### 4.1 数据收集结果

共收集文本约 **41,000字**，包括百度资讯约39,000字、备用语料库约1,700字。经过预处理（分词、去停用词、主题过滤）后，得到 **3,749个有效词语**。

### 4.2 词频统计（Top 10）

| 排名 | 词语 | 频次 |
|------|------|------|
| 1 | 人工智能 | 657 |
| 2 | 技术 | 398 |
| 3 | 发展 | 309 |
| 4 | 智能 | 135 |
| 5 | 未来 | 123 |
| 6 | 学习 | 104 |
| 7 | 模型 | 94 |
| 8 | 应用 | 92 |
| 9 | 百度 | 75 |
| 10 | 领域 | 73 |

### 4.3 可视化结果

**图1：AI主题文本数据词云图**

> （见 homework_1/output/ai_wordcloud.png）
> 词云图中"人工智能"字体最大，表明其在收集的文本中出现频率最高。其次是"技术""发展""智能""未来""学习""模型"等词，充分反映了AI领域的核心关注点。

**图2：Top 20 高频词柱状图**

> （见 homework_1/output/ai_top_words_bar.png）
> 柱状图清晰展示了排名前20的高频词及其出现次数。"人工智能"以657次遥遥领先，"技术"（398次）和"发展"（309次）紧随其后。

**图3：TF-IDF 关键词权重图**

> （见 homework_1/output/ai_keyword_weights.png）
> TF-IDF算法考虑了词频和逆文档频率，能更好地反映词语的区分性。"人工智能"的TF-IDF权重为0.8530，远高于其他词语，"技术"（0.2578）和"发展"（0.1516）位列其后。

**图4：语义类别分布饼图**

> （见 homework_1/output/ai_category_pie.png）
> 将AI相关词语按语义分为五大类别：
> - 技术概念占 55.5%（最大比例，说明收集的文本以技术讨论为主）
> - 发展趋势占 23.9%（反映对AI发展方向的高度关注）
> - 应用领域占 12.4%（涵盖医疗、金融、教育等具体应用场景）
> - 企业机构占 5.3%（涉及百度、华为等代表性企业）
> - 伦理治理占 2.8%（体现对AI安全、隐私等问题的关注）

---

## 五、总结

### 5.1 实验收获

本次作业通过对"人工智能"主题的文本数据收集与预处理，完整实践了文本信息挖掘的基础流程：

1. **数据收集方面**：学习并实践了使用 `requests` 和 `BeautifulSoup` 进行网页爬虫的技术，体会到了多源数据整合的重要性。
2. **数据预处理方面**：掌握了中文文本处理的关键步骤，包括正则清洗、jieba分词、停用词过滤和主题相关词过滤。
3. **数据可视化方面**：利用 `wordcloud` 和 `matplotlib` 生成了多种直观的可视化图表，加深了对文本数据分布特征的理解。

### 5.2 结果分析

从分析结果可以看出：
- AI领域的讨论主要集中在**技术层面**（深度学习、神经网络、算法、模型等），占比超过一半
- **发展与变革**是第二大主题，反映了社会对AI未来走向的强烈关注
- **应用落地**（医疗、金融、教育等）也是重要话题，说明AI正在从实验室走向实际应用
- **伦理治理**虽然占比较小，但数据隐私、算法安全等问题正日益受到重视

### 5.3 不足与改进

- 百度百科的反爬机制导致部分词条无法成功获取正文，后续可引入Selenium模拟浏览器或使用API接口
- 可进一步丰富数据源，如引入微博热点、知网论文摘要等渠道
- 词云图的视觉效果可通过自定义形状蒙版（如AI图标形状）进一步优化

---

## 参考文献

[1] Python官方文档. urllib.request — 用于打开URLs的可扩展库. https://docs.python.org/zh-cn/3/library/urllib.request.html

[2] Beautiful Soup Documentation. https://www.crummy.com/software/BeautifulSoup/bs4/doc/

[3] jieba中文分词. https://github.com/fxsjy/jieba

[4] WordCloud for Python. https://github.com/amueller/word_cloud

[5] Matplotlib官方文档. https://matplotlib.org/stable/

[6] 国务院. 新一代人工智能发展规划. 国发〔2017〕35号. 2017年7月.
