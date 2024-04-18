"""
@File    :   model.py
@Time    :   2024/04/16 19:24:51
@Author  :   Xiang Lei 
@Version :   1.0
@Desc    :   None
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        """
        in_channels 输入图片的通道数
        patch_size 小方块的大小
        num_patches 小方块的维度
        embed_dim 有多少个小方块
        """
        super(PatchEmbedding, self).__init__()
        # 图片 -> 切成小方块 -> 拉平
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2),
        )

        self.cls_token = nn.Parameter(
            torch.randn(size=(1, 1, embed_dim)), requires_grad=True
        )
        self.position_embedding = nn.Parameter(
            torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat([x, cls_token], dim=1)
        x = x + self.position_embedding
        x += self.dropout

        return x


class ViT(nn.Module):
    def __init__(
        self,
        in_channles,
        patch_size,
        embed_dim,
        num_patches,
        dropout,
        num_heads,
        activation,
        num_encoders,
        num_classes,
    ):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(
            in_channles, patch_size, embed_dim, num_patches, dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nheads=num_heads,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder_layers = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoders
        )
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim), nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)
        x = self.MLP(x)

        return x
