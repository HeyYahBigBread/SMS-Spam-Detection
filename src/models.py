# -*- coding: utf-8 -*-
"""深度学习模型定义：BiLSTM(注意力池化)、TextCNN、Transformer、混合模型。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    """字/词嵌入 + 双层 BiLSTM + 注意力池化(或均值池化) + MLP 分类头。"""

    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_layers=2,
                 dropout=0.3, pad_idx=0, pool="attn"):
        super().__init__()
        self.pool = pool
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(2 * hidden, 1) if pool == "attn" else None
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, L)
        mask = (x != 0).float().unsqueeze(-1)          # (B, L, 1)
        h = self.embedding(x)                          # (B, L, E)
        h, _ = self.lstm(h)                            # (B, L, 2H)
        if self.pool == "attn":
            score = self.attn(h)                       # (B, L, 1)
            score = score.masked_fill(mask == 0, -1e9)
            attn_w = torch.softmax(score, dim=1)
            h_pool = (h * attn_w).sum(dim=1)           # (B, 2H)
        else:
            denom = mask.sum(dim=1).clamp(min=1)
            h_pool = (h * mask).sum(dim=1) / denom
        return self.head(h_pool).squeeze(-1)


class TextCNN(nn.Module):
    """多卷积核 TextCNN：k=2,3,4,5 一维卷积 + 最大池化拼接。"""

    def __init__(self, vocab_size, emb_dim=128, num_filters=128,
                 kernel_sizes=(2, 3, 4, 5), dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        total = num_filters * len(kernel_sizes)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.embedding(x).transpose(1, 2)          # (B, E, L)
        feats = [F.relu(conv(h)) for conv in self.convs]
        pooled = [F.adaptive_max_pool1d(f, 1).squeeze(-1) for f in feats]
        out = torch.cat(pooled, dim=1)                 # (B, total)
        return self.head(out).squeeze(-1)


class PositionalEncoding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerCls(nn.Module):
    """嵌入 + 位置编码 + TransformerEncoder + 掩码均值池化 + MLP。"""

    def __init__(self, vocab_size, emb_dim=128, num_layers=2, num_heads=4,
                 dim_ff=256, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos = PositionalEncoding(emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        mask = x != 0                                  # (B, L) 有效位置
        h = self.pos(self.embedding(x))                # (B, L, E)
        h = self.encoder(h, src_key_padding_mask=~mask)
        denom = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        h_pool = (h * mask.float().unsqueeze(-1)).sum(dim=1) / denom
        return self.head(h_pool).squeeze(-1)


class HybridModel(nn.Module):
    """BiLSTM + TextCNN 双通道特征融合（共享嵌入层）。"""

    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_filters=128,
                 kernel_sizes=(2, 3, 4, 5), dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=2, batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.attn = nn.Linear(2 * hidden, 1)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        cnn_dim = num_filters * len(kernel_sizes)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden + cnn_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        emb = self.embedding(x)
        # BiLSTM 通道
        h, _ = self.lstm(emb)
        score = self.attn(h).masked_fill(mask == 0, -1e9)
        h_lstm = (h * torch.softmax(score, dim=1)).sum(dim=1)
        # TextCNN 通道
        c = emb.transpose(1, 2)
        feats = [F.adaptive_max_pool1d(F.relu(conv(c)), 1).squeeze(-1)
                 for conv in self.convs]
        h_cnn = torch.cat(feats, dim=1)
        return self.head(torch.cat([h_lstm, h_cnn], dim=1)).squeeze(-1)


def get_model(name, vocab_size, **kw):
    models = {
        "bilstm": BiLSTM,
        "textcnn": TextCNN,
        "transformer": TransformerCls,
        "hybrid": HybridModel,
    }
    if name not in models:
        raise ValueError(f"未知模型: {name}，可选 {list(models)}")
    cls = models[name]
    allowed = {
        "bilstm": ("emb_dim", "hidden", "num_layers", "dropout", "pad_idx",
                   "pool"),
        "textcnn": ("emb_dim", "num_filters", "kernel_sizes", "dropout",
                    "pad_idx"),
        "transformer": ("emb_dim", "num_layers", "num_heads", "dim_ff",
                        "dropout", "pad_idx"),
        "hybrid": ("emb_dim", "hidden", "num_filters", "kernel_sizes",
                   "dropout", "pad_idx"),
    }
    kw = {k: v for k, v in kw.items() if k in allowed[name]}
    return cls(vocab_size, **kw)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
