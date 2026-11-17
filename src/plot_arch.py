# -*- coding: utf-8 -*-
"""模型结构示意图绘制（用于大作业报告）。

用法：
    python src/plot_arch.py --fig-dir results/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

C_EMB = "#cfe2f3"
C_RNN = "#d9ead3"
C_CNN = "#fce5cd"
C_ATTN = "#ead1dc"
C_FC = "#fff2cc"
C_IO = "#e6e6e6"


def _box(ax, x, y, w, h, text, color, fontsize=8.5):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.03",
                         fc=color, ec="#666666", lw=1.0)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color="#444444"))


def _new_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    return ax


def draw_bilstm(path):
    ax = _new_ax((7.5, 5.0))
    _box(ax, 0.3, 7.0, 2.0, 1.6, "输入短信\n(字符/词 ID 序列)", C_IO, 9)
    _box(ax, 3.1, 7.0, 2.0, 1.6, "Embedding\n128 维", C_EMB, 9)
    _box(ax, 5.9, 7.0, 3.4, 1.6, "双向 LSTM ×2 层\n(隐藏 128×2 方向)", C_RNN, 9)
    _box(ax, 5.9, 4.2, 3.4, 1.6, "注意力池化\n(softmax 加权求和)", C_ATTN, 9)
    _box(ax, 5.9, 1.6, 3.4, 1.6, "全连接层\nDropout → 64 → 1", C_FC, 9)
    _box(ax, 5.9, 0.1, 3.4, 1.0, "sigmoid: 垃圾短信概率", C_IO, 8.5)
    _arrow(ax, 2.3, 7.8, 3.1, 7.8)
    _arrow(ax, 5.1, 7.8, 5.9, 7.8)
    _arrow(ax, 7.6, 7.0, 7.6, 5.8)
    _arrow(ax, 7.6, 4.2, 7.6, 3.2)
    _arrow(ax, 7.6, 1.6, 7.6, 1.1)
    ax.text(8.1, 3.7, "掩码\n处理", fontsize=7.5, color="#666666", ha="left")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存", path)


def draw_textcnn(path):
    ax = _new_ax((7.5, 5.2))
    _box(ax, 0.3, 6.4, 2.0, 1.8, "输入短信\n(字符/词 ID 序列)", C_IO, 9)
    _box(ax, 3.1, 6.4, 2.0, 1.8, "Embedding\n128 维", C_EMB, 9)
    for i, k in enumerate([2, 3, 4, 5]):
        y = 4.8 - i * 1.1
        _box(ax, 5.9, y, 2.6, 0.9, f"卷积核 k={k} (128 通道)", C_CNN, 8.5)
    _box(ax, 5.9, 0.5, 2.6, 1.0, "最大池化拼接", C_CNN, 8.5)
    _box(ax, 3.1, 0.5, 2.0, 1.0, "全连接层\nDropout → 64 → 1", C_FC, 8.5)
    _box(ax, 0.3, 0.5, 2.0, 1.0, "sigmoid:\n垃圾短信概率", C_IO, 8.5)
    _arrow(ax, 2.3, 7.3, 3.1, 7.3)
    _arrow(ax, 5.1, 7.3, 5.9, 7.3)
    for i in range(4):
        y = 5.25 - i * 1.1
        _arrow(ax, 5.9, y + 0.9, 5.9, 1.0)
    _arrow(ax, 5.9, 0.5, 5.1, 1.0)
    _arrow(ax, 3.1, 1.0, 2.3, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存", path)


def draw_transformer(path):
    ax = _new_ax((7.5, 5.2))
    _box(ax, 0.3, 6.4, 2.0, 1.8, "输入短信\n(字符/词 ID 序列)", C_IO, 9)
    _box(ax, 3.1, 6.4, 2.0, 1.8, "Embedding + 位置编码\n128 维", C_EMB, 9)
    _box(ax, 5.9, 5.6, 3.4, 2.6, "Transformer 编码层 ×2\n多头自注意力(4 头)\n+ 前馈网络(GELU)", C_ATTN, 9)
    _box(ax, 5.9, 2.9, 3.4, 1.6, "掩码均值池化\n(忽略 [PAD] 位置)", C_RNN, 9)
    _box(ax, 5.9, 0.5, 3.4, 1.6, "全连接层\nDropout → 64 → 1", C_FC, 9)
    _arrow(ax, 2.3, 7.3, 3.1, 7.3)
    _arrow(ax, 5.1, 7.3, 5.9, 7.3)
    _arrow(ax, 7.6, 5.6, 7.6, 4.5)
    _arrow(ax, 7.6, 2.9, 7.6, 2.1)
    ax.text(7.7, 0.1, "sigmoid 输出", fontsize=8, color="#444444")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存", path)


def draw_hybrid(path):
    ax = _new_ax((9.5, 5.6))
    _box(ax, 0.3, 6.2, 2.0, 1.8, "输入短信\n(字符/词 ID 序列)", C_IO, 9)
    _box(ax, 3.0, 6.2, 2.0, 1.8, "共享 Embedding\n128 维", C_EMB, 9)
    _box(ax, 5.7, 7.6, 3.3, 1.8, "双向 LSTM ×2 层", C_RNN, 9)
    _box(ax, 5.7, 5.4, 3.3, 1.4, "注意力池化", C_ATTN, 9)
    _box(ax, 5.7, 3.2, 3.3, 1.4, "多核卷积 (k=2,3,4,5)\n+ 最大池化", C_CNN, 9)
    _box(ax, 5.7, 0.9, 3.3, 1.5, "特征拼接 (256+512 维)", C_FC, 9)
    _box(ax, 2.4, 0.9, 2.2, 1.5, "全连接层\nDropout → 64 → 1", C_FC, 9)
    _box(ax, 0.3, 0.9, 1.5, 1.5, "sigmoid\n输出", C_IO, 8.5)
    _arrow(ax, 2.3, 7.1, 3.0, 7.1)
    _arrow(ax, 5.0, 7.1, 5.7, 7.1)
    _arrow(ax, 7.35, 7.6, 7.35, 6.8)
    _arrow(ax, 5.7, 8.5, 5.0, 7.1)  # 共享嵌入分流示意
    _arrow(ax, 7.35, 5.4, 7.35, 4.6)
    _arrow(ax, 7.35, 3.2, 7.35, 2.4)
    _arrow(ax, 5.7, 1.65, 4.6, 1.65)
    _arrow(ax, 2.4, 1.65, 1.8, 1.65)
    ax.text(5.15, 8.6, "双通道特征提取", fontsize=8, color="#666666")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    draw_bilstm(os.path.join(args.fig_dir, "arch_bilstm.png"))
    draw_textcnn(os.path.join(args.fig_dir, "arch_textcnn.png"))
    draw_transformer(os.path.join(args.fig_dir, "arch_transformer.png"))
    draw_hybrid(os.path.join(args.fig_dir, "arch_hybrid.png"))


if __name__ == "__main__":
    main()
