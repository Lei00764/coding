"""
@File    :   attn.py
@Time    :   2024/04/11 10:13:31
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim=None, num_heads=1):
        super(CrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_heads = num_heads

        """
        Q 与 K 的维度必须相等
        V 的维度没有限制
        """

        self.query = nn.Linear(
            hidden_dim, embedding_dim, bias=False
        )  # 我问了一个问题，jifa 的生活是怎么样的

        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(
                hidden_dim, embedding_dim, bias=False
            )  # 我朋友跟我说了一下 jifa 的生活（比较简短）
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 详细的版本
        else:
            self.self_attn = False
            self.key = nn.Linear(
                context_dim, embedding_dim, bias=False
            )  # 导演跟我说了一下 jifa 的生活（相对于朋友要详细）
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)  # 详细的版本

    def forward(self, tokens, context=None):
        if self.self_attn:
            Q = self.query(tokens)
            K = self.key(tokens)
            V = self.value(tokens)

        else:
            Q = self.query(tokens)
            K = self.key(context)
            V = self.value(context)

        score_matrix = torch.einsum("BTH, BSH -> BTS", Q, K)
        score_matrix = score_matrix / np.sqrt(self.embedding_dim)
        score_matrix = F.softmax(score_matrix, dim=-1)
        ctx_vecs = torch.einsum("BTS, BSH -> BTH", score_matrix, V)
        return ctx_vecs


if __name__ == "__main__":
    batch_size = 1
    seq_len = 4  # 4 个英文单词
    hidden_dim = 64  # 为每个英文单词创建多少维度的向量来表示
    embedding_dim = 32  # 少于 hidden_dim 的维度
    context_seq_len = 8  # 要翻译成 8 个中文单词
    context_dim = 128  # 为每个中文单词创建多少维度的向量来表示

    model = CrossAttention(embedding_dim, hidden_dim, context_dim)

    tokens = torch.randn(batch_size, seq_len, hidden_dim)  # 4 个英文单词
    context = torch.randn(
        batch_size, context_seq_len, context_dim
    )  # 导演告诉我 8 个中文单词

    output = model(tokens, context)

    print(tokens.shape)
    print(context.shape)
    print(output.shape)  # [batch_size, seq_len(tokens), hidden_dim]
