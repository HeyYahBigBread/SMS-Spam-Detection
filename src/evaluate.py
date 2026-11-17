# -*- coding: utf-8 -*-
"""统一评估与可视化：汇总各模型结果，输出对比表、ROC 曲线、
混淆矩阵、训练曲线，并做错误样例分析。

用法：
    python src/evaluate.py --results-dir results --fig-dir results/figures
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]

ABLATION_SUFFIXES = ("_meanpool", "_noweight")


def is_ablation(row):
    """消融实验的结果不进入主对比表。"""
    tag = row.get("tag", "")
    return any(tag.endswith(s) for s in ABLATION_SUFFIXES)


def collect_results(results_dir):
    """收集 ML 与 DL 的全部测试集结果。"""
    rows = []
    for path in glob.glob(os.path.join(results_dir, "ml", "*_test.json")):
        rows.append(load_json(path))
    for path in glob.glob(os.path.join(results_dir, "models", "*_test.json")):
        rows.append(load_json(path))
    return rows


def model_display_name(row):
    lvl = {"char": "字级", "word": "词级"}.get(row.get("level"), row.get("level"))
    if "params" in row:  # 深度学习模型
        m = {"bilstm": "BiLSTM", "textcnn": "TextCNN",
             "transformer": "Transformer", "hybrid": "BiLSTM+TextCNN"}[row["model"]]
        return f"{m}({lvl})"
    m = {"MultinomialNB": "朴素贝叶斯", "LogisticRegression": "逻辑回归",
         "LinearSVC": "线性SVM"}.get(row.get("model"), row.get("model"))
    return f"{m}({lvl})"


def build_comparison_table(rows):
    """生成对比表 DataFrame。"""
    data = []
    for r in rows:
        t = r["test"]
        data.append({
            "模型": model_display_name(r),
            "Accuracy": t["accuracy"], "Precision": t["precision"],
            "Recall": t["recall"], "F1": t["f1"], "AUC": t["auc"],
            "参数量/特征数": r.get("params", r.get("n_features", None)),
            "训练耗时(s)": r.get("train_time_s", r.get("fit_time_s", None)),
            "推理耗时(s)": r.get("infer_time_s", None),
        })
    df = pd.DataFrame(data)
    df = df.sort_values("F1", ascending=False).reset_index(drop=True)
    return df


def plot_roc_curves(rows, fig_dir):
    """所有模型的 ROC 曲线。"""
    plt.figure(figsize=(7, 6))
    for r in rows:
        if "params" in r:
            npz = os.path.join("results", "models",
                               f"{r['model']}_{r['level']}_test_pred.npz")
        else:
            npz = os.path.join("results", "ml",
                               f"{r.get('tag', r['level'] + '_' + r['model'])}_test_pred.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz)
        y, s = d["y"], d["score"]
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, lw=1.6,
                 label=f"{model_display_name(r)} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC 曲线对比")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_curves.png"), dpi=200)
    plt.close()
    print("已保存 results/figures/roc_curves.png")


def plot_confusion_matrices(rows, fig_dir, top_n=4):
    """F1 最高的 top_n 个模型的混淆矩阵。"""
    rows_sorted = sorted(rows, key=lambda r: r["test"]["f1"], reverse=True)
    n = min(top_n, len(rows_sorted))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, rows_sorted[:n]):
        tn, fp, fn, tp = r["test"]["tn_fp_fn_tp"]
        cm = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["正常", "垃圾"]); ax.set_yticklabels(["正常", "垃圾"])
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=11, color="white" if cm[i, j] > cm.max() / 2
                        else "black")
        ax.set_title(f"{model_display_name(r)}\nF1={r['test']['f1']:.4f}",
                     fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices.png"), dpi=200)
    plt.close()
    print("已保存 results/figures/confusion_matrices.png")


def plot_training_curves(fig_dir):
    """深度学习模型的训练曲线（val F1 随 epoch 变化）。"""
    paths = sorted(glob.glob(os.path.join("results", "models",
                                          "*_history.csv")))
    if not paths:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for p in paths:
        df = pd.read_csv(p)
        name = os.path.basename(p).replace("_history.csv", "")
        axes[0].plot(df["epoch"], df["val_acc"], marker="o", ms=3, label=name)
        axes[1].plot(df["epoch"], df["val_f1"], marker="o", ms=3, label=name)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("验证集 Accuracy")
    axes[0].set_title("验证集 Accuracy 曲线")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("验证集 F1")
    axes[1].set_title("验证集 F1 曲线")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_curves.png"), dpi=200)
    plt.close()
    print("已保存 results/figures/training_curves.png")


def error_analysis(rows, fig_dir, n=10):
    """输出最优模型的错误样例（假阳性/假阴性各 n 条）。"""
    best = max(rows, key=lambda r: r["test"]["f1"])
    if "params" not in best:  # 只对深度学习模型做，ML 也可做但不保存文本
        return
    test = pd.read_csv("data/processed/test.csv")
    npz = os.path.join("results", "models",
                       f"{best['model']}_{best['level']}_test_pred.npz")
    d = np.load(npz)
    y, pred = d["y"], d["pred"]
    fp_idx = np.where((y == 0) & (pred == 1))[0]
    fn_idx = np.where((y == 1) & (pred == 0))[0]
    lines = [f"模型: {model_display_name(best)}",
             f"假阳性(正常误判为垃圾) {len(fp_idx)} 条，"
             f"假阴性(垃圾漏判) {len(fn_idx)} 条", ""]
    lines.append("== 假阳性样例（正常短信被误判） ==")
    for i in fp_idx[:n]:
        lines.append(f"  [{i}] {test.iloc[i]['text'][:80]}")
    lines.append("")
    lines.append("== 假阴性样例（垃圾短信被漏判） ==")
    for i in fn_idx[:n]:
        lines.append(f"  [{i}] {test.iloc[i]['text'][:80]}")
    with open(os.path.join(fig_dir, "error_analysis.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已保存 results/figures/error_analysis.txt "
          f"(FP={len(fp_idx)}, FN={len(fn_idx)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    rows = collect_results(args.results_dir)
    if not rows:
        print("未找到结果文件，请先运行训练脚本")
        return
    rows = [r for r in rows if not is_ablation(r)]
    print(f"共收集 {len(rows)} 个模型结果（不含消融实验）")

    df = build_comparison_table(rows)
    df.to_csv(os.path.join(args.fig_dir, "comparison.csv"),
              index=False, encoding="utf-8-sig")
    with pd.option_context("display.float_format",
                           lambda v: f"{v:.4f}"):
        print("\n========== 模型对比表 ==========")
        print(df.to_string(index=False))
    print("已保存 results/figures/comparison.csv")

    plot_roc_curves(rows, args.fig_dir)
    plot_confusion_matrices(rows, args.fig_dir)
    plot_training_curves(args.fig_dir)
    error_analysis(rows, args.fig_dir)


if __name__ == "__main__":
    main()
