"""
@Project ：coding 
@File    ：MultiHeadAttention.py
@Content ： 
@Author  ：Xiang Lei
@Email   ：xiang.lei.se@foxmail.com
@Date    ：4/21/2024 3:02 PM 
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """

        :param embed_dim: 词向量维度
        :param num_heads:
        """
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)  # 在最后一个维度上进行 softmax
        self.num_heads = num_heads

    def forward(self, x):
        """

        :param x: [batch_size, seq_len, embed_dim]
        :return:
        """
        B, L, H = x.shape
        assert H % self.num_heads == 0  # 确保 embed_dim 可以被 num_heads 整除
        head_dim = H // self.num_heads  # 每个 head 的维度

        # 将 x 转换为 Q, K, V
        Q = self.W_Q(x).view(B, L, self.num_heads, head_dim)
        K = self.W_K(x).view(B, L, self.num_heads, head_dim)
        V = self.W_V(x).view(B, L, self.num_heads, head_dim)

        # 计算注意力分数
        attention_score = Q @ K.transpose(-2, -1) / head_dim**0.5
        attention_score = self.softmax(attention_score)

        # 计算注意力值
        attention = attention_score @ V

        return attention.view(B, L, H)


if __name__ == "__main__":
    x = torch.rand(2, 10, 512)  # [batch_size, seq_len, embed_dim]
    multi_head_attention = MultiHeadAttention(512, 8)
    output = multi_head_attention(x)
    print(output.shape)  # torch.Size([2, 10, 512])
