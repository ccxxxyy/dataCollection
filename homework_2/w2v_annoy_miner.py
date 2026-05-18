# -*- coding: utf-8 -*-
"""
词向量训练与近似近邻检索。
优先使用 Gensim Word2Vec + Annoy；
"""
import numpy as np

try:
    from annoy import AnnoyIndex

    _HAS_ANNOY = True
except ImportError:
    AnnoyIndex = None
    _HAS_ANNOY = False

try:
    from gensim.models import Word2Vec

    _HAS_GENSIM = True
except ImportError:
    Word2Vec = None
    _HAS_GENSIM = False


class EmbeddingModel:
    """统一封装：word2idx、词表顺序、行已 L2 归一化的向量矩阵 (n, d)"""

    def __init__(self, vectors, words):
        self.words = list(words)
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        m = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = (m / norms).astype(np.float32)

    def vector_for(self, word):
        i = self.word2idx.get(word)
        if i is None:
            return None
        return self.matrix[i]


def train_word2vec(tokenized_docs, vector_size, window, min_count, epochs, workers):
    if not _HAS_GENSIM:
        raise RuntimeError("gensim 未安装")
    return Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=42,
    )


def _gensim_to_embedding(model):
    kv = model.wv
    words = list(kv.index_to_key)
    mat = np.stack([kv[w] for w in words], axis=0)
    return EmbeddingModel(mat, words)


def _train_svd_ppmi(tokenized_docs, vector_size, window, max_vocab):
    """小规模语料的纯 NumPy 词向量：共现 + PPMI + SVD。"""
    from collections import Counter

    fc = Counter()
    for sent in tokenized_docs:
        fc.update(sent)
    vocab = [w for w, _ in fc.most_common(max_vocab)]
    if len(vocab) < 10:
        raise ValueError("词表过小，无法训练词向量")
    w2i = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    co = np.zeros((n, n), dtype=np.float64)
    for sent in tokenized_docs:
        idxs = [w2i[t] for t in sent if t in w2i]
        for i, a in enumerate(idxs):
            lo = max(0, i - window)
            hi = min(len(idxs), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                b = idxs[j]
                co[a, b] += 1.0
    total = co.sum()
    if total <= 0:
        raise ValueError("共现矩阵为空")
    pij = co / total
    pi = pij.sum(axis=1)
    pj = pij.sum(axis=0)
    denom = np.outer(pi, pj) + 1e-12
    ratio = pij / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log(ratio + 1e-12)
    ppmi = np.maximum(0.0, ppmi)
    ppmi[pij <= 0] = 0.0
    k = min(vector_size, n - 1, 128)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    emb = u[:, :k] * np.sqrt(s[:k] + 1e-9)
    return EmbeddingModel(emb.astype(np.float32), vocab)


def train_embedding_model(
    tokenized_docs,
    vector_size,
    window,
    min_count,
    epochs,
    workers,
    max_vocab_np=1800,
):
    if _HAS_GENSIM:
        m = train_word2vec(
            tokenized_docs, vector_size, window, min_count, epochs, workers
        )
        print("[词向量] 使用 Gensim Word2Vec")
        return _gensim_to_embedding(m), m
    print("[词向量] 未检测到 Gensim，使用 NumPy 共现矩阵 + SVD（作业环境兼容）")
    emb = _train_svd_ppmi(tokenized_docs, vector_size, window, max_vocab_np)
    return emb, None


def knn_candidates_from_annoy(embed_model, gensim_model, domain_words, k, n_trees):
    """使用 Annoy（需 gensim 安装的同一环境里通常也能装 annoy；否则外层跳过）。"""
    if not _HAS_ANNOY or gensim_model is None:
        return None
    kv = gensim_model.wv
    words_list = list(kv.index_to_key)
    dim = kv.vector_size
    index = AnnoyIndex(dim, "angular")
    for i, w in enumerate(words_list):
        v = kv[w].astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        index.add_item(i, v)
    index.build(n_trees)
    word2idx = {w: i for i, w in enumerate(words_list)}
    domain = set(domain_words)
    candidates = set()
    for w in domain:
        if w not in word2idx:
            continue
        idx = word2idx[w]
        n_ids = index.get_nns_by_item(idx, k + 8, include_distances=False)
        taken = 0
        for nid in n_ids:
            if nid == idx:
                continue
            nw = words_list[nid]
            if nw not in domain:
                candidates.add(nw)
            taken += 1
            if taken >= k:
                break
    return candidates


def knn_candidates_numpy(embed_model, domain_words, k):
    """全量余弦相似度：对中小词表足够快。"""
    domain = set(domain_words)
    M = embed_model.matrix
    words = embed_model.words
    candidates = set()
    for w in domain:
        i = embed_model.word2idx.get(w)
        if i is None:
            continue
        v = M[i]
        sims = M @ v
        order = np.argsort(-sims)
        taken = 0
        for j in order:
            if j == i:
                continue
            if words[j] not in domain:
                candidates.add(words[j])
            taken += 1
            if taken >= k:
                break
    return candidates


def knn_candidates(
    embed_model, gensim_model, domain_words, k, n_trees
):
    c = knn_candidates_from_annoy(embed_model, gensim_model, domain_words, k, n_trees)
    if c is not None:
        return c
    return knn_candidates_numpy(embed_model, domain_words, k)
