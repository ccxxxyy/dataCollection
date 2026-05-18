# 作业二 定制化词典

## 一、作业要求

1. 收集、整理并产生一份个人（文本）**定制化词典**，内容、方向、体裁、格式、创意不限。
2. 使用定制化词典对语料进行**分词**，并进行展示。
3. 加分方向：有趣的数据与创意、有效的新词评估机制、Matplotlib / Jupyter 等可视化与规范写作；参考需列入文献。

文档推荐的技术路线：**结巴分词** → **词向量（如 Word2Vec）** → **Annoy 向量索引** → 对词典中词汇做 **K 近邻** 得到新词候选 → 用 **TF、DF** 过滤后迭代扩充词典，直至稳定。

---

## 二、选题与总体设计

本作业构建**两条垂直领域**的定制化词典并做对照：

| 领域 | 初始种子词文件 | 说明 |
|------|----------------|------|
| 人工智能 | `seed_terms_ai.txt` | 以大模型、机器学习、NLP 等术语为起点 |
| 财经证券 | `seed_terms_finance.txt` | 以财报、基金、货币政策等术语为起点 |

**通用词典**沿用结巴内置词典；**领域增量**通过用户词典文件（`output/user_dict_*.txt`）注入，使长词、专有名词优先整词切分。

整体流水线：

```
采集语料 → 写入用户词典（种子）→ 分词 → 训练词向量 → K 近邻扩展候选
    → TF/DF 过滤 → 写入用户词典 → … 直到无新词或达到轮次上限 → 可视化与例句对比
```

---

## 三、数据来源与语料构建

- **百度百科**：尝试爬取 AI / 财经相关词条正文（与作业一相同的 `requests` + `BeautifulSoup` 方式；若遇反爬或网络限制则正文可能为空）。
- **百度资讯**：关键词如「人工智能产业发展」「A股市场行情」抓取标题与摘要片段。
- **本地补充**：若存在 `homework_1/ai_corpus.txt`，一并合并，保证离线可复现。
- **保底短文本**：当在线语料过少时，追加内置短综述段落，避免流水线空跑。

原始合并语料保存为 `output/mixed_raw_corpus.txt`。

---

## 四、实现要点

### 4.1 分词与用户词典

- 使用 `jieba.load_userdict` 加载 `用户词 词频` 格式文件。
- 文本经简单清洗（保留中文、去空白）后分词，并用哈工大停用词表 `hit_stopwords.txt` 过滤。

### 4.2 词向量与近邻检索

- **优先**：安装成功时使用 **Gensim** `Word2Vec` 与 **Annoy** 索引，与作业说明一致。
- **兼容**：在当前常见的仅带 Python 3.14、缺少 MSVC 编译链的环境中，Gensim/Annoy 可能无法从源码编译安装。此时代码**自动回退**为 **NumPy 共现矩阵 + PPMI + SVD** 得到稠密词向量，并用矩阵乘法做 **余弦 Top-K**，保证流程可运行、报告可撰写。回退时在控制台打印 `[词向量] 未检测到 Gensim，使用 NumPy …`。

### 4.3 新词评估（TF / DF）

- 将语料按**空行分段**视为多篇「文档」。
- 对 K 近邻产生且不在当前领域词典内的候选词，统计：
  - **TF**：全语料词频；
  - **DF**：至少出现在多少个文档中。
- 默认阈值：`MIN_TF = 3`，`MIN_DF = 2`（可在 `config.py` 调整）。
- 每轮最多合并 32 个新词（控制噪声），迭代上限见 `MAX_ITERATIONS`。

### 4.4 展示与对比

- `output/segment_compare_demo.txt`：同一批例句在 **AI 用户词典** 与 **财经用户词典** 下的分词结果对照。
- `output/dict_growth.png`：两轮定制词典规模随迭代变化。
- `output/token_freq_ai_dict.png`、`token_freq_finance_dict.png`：分词后高频词分布。

运行入口：`python homework_2/main.py`（

Jupyter 示意：`homework2.ipynb` 中可逐步执行采集、分词与读图。

---

## 五、文件说明

| 路径 | 作用 |
|------|------|
| `config.py` | 路径与超参数 |
| `corpus_collector.py` | 语料采集与保存 |
| `dict_manager.py` | 种子词与用户词典读写 |
| `segmentation.py` | 结巴分词与停用词 |
| `text_process.py` | 文档切分、TF/DF |
| `w2v_annoy_miner.py` | Word2Vec/Annoy 与 NumPy 回退 |
| `pipeline.py` | 迭代扩词主逻辑 |
| `visualize.py` | Matplotlib 出图 |
| `main.py` | 一键跑通全流程 |
| `requirements.txt` | 依赖列表（Gensim/Annoy 为可选增强） |

---

## 六、结果小结

- 在测试运行中，资讯与本地语料可支撑词向量训练与新词迭代；AI 与财经两套种子在不同近邻与 TF/DF 规则下形成不同规模的**终态用户词典**（见 `output/dictionary_stats.txt`）。
- 分词对比文件显示：在引入领域长词（如「净资产收益率」「大语言模型」）后，整词切分更稳定；部分词（如「逆回购」）可通过继续扩充种子词表进一步优化。

---

## 七、参考文献

1. Gensim, Topic Modelling for humans, [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/), Accessed 2021-03-30.
2. Annoy (Approximate Nearest Neighbors Oh Yeah), [https://github.com/spotify/annoy](https://github.com/spotify/annoy), Accessed 2021-03-30.
3. 结巴中文分词, [https://github.com/fxsjy/jieba](https://github.com/fxsjy/jieba).
