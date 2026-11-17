# -*- coding: utf-8 -*-
"""数据集统计可视化：标签分布饼图、文本长度分布直方图。

用法：
    python src/plot_data_stats.py --data-dir data/processed --fig-dir results/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    total = len(train) + len(val) + len(test)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 标签分布
    spam = int((train["label"] == 1).sum()) + int((val["label"] == 1).sum()) \
        + int((test["label"] == 1).sum())
    axes[0].pie([total - spam, spam],
                labels=[f"正常短信 {total - spam:,}",
                        f"垃圾短信 {spam:,}"],
                colors=["#9ecae1", "#fc9272"], autopct="%.1f%%",
                startangle=90, textprops={"fontsize": 11})
    axes[0].set_title(f"标签分布（共 {total:,} 条）")

    # 长度分布（按划分）
    for name, part in [("train", train), ("val", val), ("test", test)]:
        axes[1].hist(part["text"].str.len(), bins=50, range=(0, 150),
                     alpha=0.5, density=True, label=f"{name}({len(part):,})")
    axes[1].set_xlabel("短信长度（字符数）")
    axes[1].set_ylabel("密度")
    axes[1].set_title("短信文本长度分布")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_dir, "data_stats.png"), dpi=200)
    plt.close()
    print("已保存 results/figures/data_stats.png")


if __name__ == "__main__":
    main()
