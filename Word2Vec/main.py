"""
@File    :   main.py
@Time    :   2024/04/26 16:02:33
@Author  :   Xiang Lei
@Version :   1.0
@Desc    :   None
"""

"""
三大参数：index2word, word2index, word2vec

为什么需要这三个参数？
1. word2index: 通过 word 找到 index
2. index2word: 通过 index 找到 word
3. word2onehot: 通过 word 找到 onehot
"""

# 超参数
# 词向量维度 embedding_dim = 50（自己设置）
# 词汇表大小 vocab_size = 5000（通过统计词频得到）

import os
import pickle
import jieba

from tqdm import tqdm
import numpy as np
import pandas as pd


def load_stop_words(file_path):
    """
    加载停用词
    """
    with open(file_path, "r", encoding="utf-8") as f:
        stop_words = f.read().split("\n")
    return stop_words


def cut_words(file_path):
    stop_words = load_stop_words(".\src\stopwords.txt")

    result = []
    all_data = pd.read_csv(file_path, encoding="gbk", names=["data"])["data"]
    for words in all_data:
        c_words = jieba.cut(words)  # 分词
        c_words = [word for word in c_words if word not in stop_words]  # 去停用词
        result.append(c_words)

    return result


def get_dict(data):
    """
    构建 index2word, word2index, word2onehot
    """
    index2word = []
    for words in data:
        for word in words:
            if word not in index2word:
                index2word.append(word)

    word2index = {
        word: index for index, word in enumerate(index2word)
    }  # 通过 word 找到 index

    word_size = len(index2word)
    word2onehot = {}
    for word, index in word2index.items():
        one_hot = np.zeros((1, word_size))
        one_hot[0, index] = 1
        word2onehot[word] = one_hot

    return index2word, word2index, word2onehot


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def train():
    data = cut_words(".\src\data.csv")
    index2word, word2index, word2onehot = get_dict(data)

    word_size = len(index2word)
    embedding_dim = 50
    lr = 0.01
    epoch = 100
    n_gram = 3  # 左边和右边各取3个词
    w1 = np.random.normal(-1, 1, size=(word_size, embedding_dim))  # -1到1之间的随机数
    w2 = np.random.normal(-1, 1, size=(embedding_dim, word_size))

    # 伪逆矩阵，为什么要有两个矩阵？
    # 矩阵求逆的复杂度高，用随机梯度下降法求解复杂度低

    for e in range(epoch):
        for words in tqdm(data):
            for n_index, now_word in enumerate(words):
                now_word_onehot = word2onehot[now_word]
                other_words = (
                        words[max(0, n_index - n_gram): n_index]
                        + words[n_index + 1: n_index + n_gram + 1]
                )
                for other_word in other_words:
                    other_word_onehot = word2onehot[other_word]

                    hidden = now_word_onehot @ w1
                    p = hidden @ w2
                    s = softmax(p)
                    # loss = -np.sum(other_word_onehot * np.log(s))  # 算损失意义不大

                    """
                    矩阵求导
                    e.g.
                    A @ B = C
                    如果 delta_C = G
                    则 delta_A = G @ B.T
                    delta_B = A.T @ G
                    """
                    # 反向传播
                    G2 = s - other_word_onehot  # 误差
                    delta_w2 = hidden.T @ G2  # w2 的梯度
                    G1 = G2 @ w2.T
                    delta_w1 = now_word_onehot.T @ G1

                    # 更新参数
                    w1 -= lr * delta_w1
                    w2 -= lr * delta_w2

    # 保存参数
    with open("word2vec.pkl", "wb") as f:
        pickle.dump((w1, word2index, index2word, w2), f)


if __name__ == "__main__":
    train()
