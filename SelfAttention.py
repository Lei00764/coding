"""
@File    :   SelfAttention.py
@Time    :   2024/10/22 19:35:07
@Author  :   Xiang Lei 
@Email   :   xiang.lei.se@foxmail.com
@Version :   1.0
@Desc    :   None
"""

import math
import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        """
        hidden_dim:
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # linear projection for q, k, v
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Q, K, V: (batch_size, seq_len, hidden_dim)
        # 忽略 batch_size 维度，看成矩阵维度

        attention_value = torch.matmul(
            Q, K.transpose(-2, -1)
        )  # attention_value: (batch_size, seq_len, seq_len)
        attention_weight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim), dim=-1
        )

        # Why divide sqrt(hidden_dim)?
        # Because the variance of the attention score is too high, which will make the softmax function saturated.

        output = torch.matmul(attention_weight, V)

        return output  # output: (batch_size, seq_len, hidden_dim)


# How to improve the performance of self-attention?
# 1. 效率优化 SelfAttentionV2（模型较小时，可加速）
# 2. dropout, attention_mask, and output 矩阵映射


class SelfAttentionV2(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x):
        QKV = self.proj(x)  # QKV: (batch_size, seq_len, hidden_dim * 3)
        Q, K, V = QKV.split(self.hidden_dim, dim=-1)

        # then do the same thing as SelfAttentionV1


class SelfAttentionV3(nn.Module):
    def __init__(
        self, hidden_dim: int = 728, dropout_rate: float = 0.1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)  # dropout layer

        # selected output矩阵
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        QKV = self.proj(x)
        Q, K, V = QKV.split(self.hidden_dim, dim=-1)

        attention_weight = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.hidden_dim
        )

        # attention_mask: (batch_size, seq_len, seq_len)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask, float("-inf")
            )  # if attention_mask is True, then fill with -inf

        # mask first, then softmax
        attention_weight = torch.softmax(attention_weight, dim=-1)
        print(attention_weight)
        # dropout
        attention_weight = self.dropout(attention_weight)

        output = torch.matmul(attention_weight, V)
        output = self.output_proj(output)

        return output


# 面试写法
class SelfAttentionInterview(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.dim = dim

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout_rate)  # random dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, dim)
        mask: (batch_size, seq_len, seq_len)
        """
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        attention_weight = Q @ K.transpose(-2, -1) / math.sqrt(self.dim)

        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask, float("-inf"))
        # attention_weight: (batch_size, seq_len, seq_len)
        attention_weight = torch.softmax(
            attention_weight, dim=-1
        )  # softmax 需要指定维度
        attention_weight = self.dropout(attention_weight)

        output = attention_weight @ V

        return output


class MultiHeadSelfAttention(nn.module):
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.query_proj = nn.Linear(dim, dim)  # (dim, dim=head_dim * num_heads)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.output_proj = nn.Linear(dim, dim)  # selected

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, dim)
        mask: (batch_size, seq_len, seq_len)
        """
        batch_size = x.shape[0]

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # transpose(1, 2) 把 num_head 移到前面，这样可以专心处理后两个维度
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_weight = (
            Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        )  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            attention_weight = attention_weight.masked_fill(mask, float("-inf"))

        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.dropout(attention_weight)

        output = attention_weight @ V  # (batch_size, num_heads, seq_len, head_dim)
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        )  # (batch_size, seq_len, dim)
        output = self.output_proj(output)

        return output


if __name__ == "__main__":
    x = torch.randn(3, 4, 2)  # (batch_size, seq_len, hidden_dim)
    # the shape of mask need to be (batch_size, seq_len, seq_len) because attention_weight: (batch_size, seq_len, seq_len)
    mask = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=torch.bool)
    mask = mask.unsqueeze(1).expand(-1, 4, -1)  # (batch_size, seq_len, seq_len)
    print(mask.shape)
    model = SelfAttentionV3(2)
    output = model(x, mask)
    print(output.shape)
