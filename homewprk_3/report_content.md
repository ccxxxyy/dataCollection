# 作业三 文档隐含义/主题挖掘与推荐系统

## 一、作业要求理解

本作业以 **LSA/LSI/LDA 等降维技术** 为出发点，设计一套基于降维的文本推荐与可视化框架，流程为：

```
数据集处理 → 数据降维 → 降维后表示 → KNN 推荐 + 可视化（PCA / t-SNE / UMAP）
```

要求使用 Jupyter Notebook 展示，推荐复用作业一、作业二数据，并配合图表说明算法效果。

## 二、选题与数据

| 数据集 | 来源 | 说明 |
|--------|------|------|
| 中文 AI 语料 | `homework_1/ai_corpus.txt` + `homework_2/output/mixed_raw_corpus.txt` | 延续前两次作业的人工智能主题 |
| 20 Newsgroups 子集 | `sklearn.datasets` | 英文对照实验，4 个新闻组各取若干文档 |

## 三、理论与实现

### 3.1 LSA/LSI（SVD）

LSI/LSA 基于奇异值分解：对文档-词项矩阵 \(A\) 有 \(A = U \Sigma V^T\)。取前 \(k\) 个奇异值可得到低秩逼近，保留主要语义结构。

- **优势**：高效、简单、可解释。
- **劣势**：难以处理一词多义；对新文档需重新映射。

实现：`numpy.linalg.svd` 演示理论；`sklearn.decomposition.TruncatedSVD` 在 TF-IDF 稀疏矩阵上训练 LSI。

### 3.2 LDA

LDA 假设文档由 \(k\) 个隐主题混合生成，词项在主题上服从多项分布，通过 EM 估计文档-主题、主题-词项分布。

实现：`sklearn.decomposition.LatentDirichletAllocation` + 词频矩阵。

### 3.3 可视化

将 LSI 文档向量投影到二维：

- **PCA**：线性主成分，快速基线。
- **t-SNE**：保留局部邻域结构。
- **UMAP**：基于黎曼流形假设的非线性降维。

### 3.4 KNN 推荐

在 LSI 文档嵌入空间建立余弦距离 KNN 索引：给定查询文档，返回语义最接近的 \(K\) 篇文档；并支持关键词检索作为补充。

## 四、运行方式

```bash
cd homewprk_3
pip install -r requirements.txt
python main.py
```

或在仓库根目录打开 `homework3.ipynb` 逐步运行。

## 五、关于 Gensim

作业文档推荐 Gensim 框架。当前环境为 Python 3.14，Gensim 暂无预编译 wheel 且本地缺少 MSVC 编译工具。本实现使用 **sklearn + numpy** 完成等价的 TF-IDF → LSI/LDA → 嵌入 → KNN 流程，算法思想与作业要求一致。

## 六、参考文献

1. Gensim, Topic Modelling for humans, https://radimrehurek.com/gensim/
2. Annoy (Approximate Nearest Neighbors Oh Yeah), https://github.com/spotify/annoy
3. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, https://umap-learn.readthedocs.io/en/latest/
4. scikit-learn User Guide — Decomposition and Manifold learning
