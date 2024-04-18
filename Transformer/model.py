"""
@File    :   model.py
@Time    :   2024/04/14 17:02:38
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   None
"""

import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbedding, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        # 自动学习每个词的词向量
        self.embedding = nn.Embedding(
            vocab_size, d_model
        )  # vocab_size 词表大小，d_model 词向量维度

    def forward(self, x):
        # 为什么要乘以 math.sqrt(self.d_model)？
        # 放大特征
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        dropout: 丢弃率，为了防止过拟合
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # 创建一个大小为 (seq_len, d_model) 的矩阵
        pe = torch.zeros(seq_len, d_model)
        # 创建一个大小为 (seq_len) 的位置向量
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码
        div_term = torch.exp(
            torch.arrange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 偶数位置：sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置：cos
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch 维度 (1, seq_len, d_model)

        self.register_buffer("pe", pe)  # 将 pe 加入到模型的缓冲区中

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
