# -*- coding: utf-8 -*-
"""数据预处理：清洗、去重、统计、分层划分 train/val/test、构建字/词词表。

用法：
    python src/preprocess.py --input data/raw/labeled_sms.txt \
        --outdir data/processed --seed 42
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_raw, clean_text, char_tokenize, word_tokenize,
                    build_vocab, save_json, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/labeled_sms.txt")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--char-max-len", type=int, default=100)
    ap.add_argument("--word-max-len", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- 1. 读取与清洗 ----------
    print("[1/5] 读取并清洗数据 ...")
    df = load_raw(args.input)
    n_raw = len(df)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 2].reset_index(drop=True)

    # 文本去重：同一文本若标签冲突则丢弃该文本
    dup = df.groupby("text")["label"].nunique()
    conflict = set(dup[dup > 1].index)
    df = df[~df["text"].isin(conflict)]
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    n_clean = len(df)
    spam = int((df["label"] == 1).sum())
    print(f"    原始 {n_raw} 条 -> 清洗去重后 {n_clean} 条 "
          f"(垃圾短信 {spam} 条, {spam / n_clean:.2%})")

    # ---------- 2. 分层划分 ----------
    print("[2/5] 分层划分 train/val/test ...")
    from sklearn.model_selection import train_test_split
    train, rest = train_test_split(
        df, test_size=args.val_ratio + args.test_ratio,
        stratify=df["label"], random_state=args.seed)
    val_frac = args.val_ratio / (args.val_ratio + args.test_ratio)
    val, test = train_test_split(
        rest, test_size=1 - val_frac,
        stratify=rest["label"], random_state=args.seed)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(os.path.join(args.outdir, f"{name}.csv"),
                    index=False, encoding="utf-8")

    len_stats = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        n = len(part)
        s = int((part["label"] == 1).sum())
        len_stats[name] = {
            "n": n, "spam": s, "spam_ratio": round(s / n, 4),
            "mean_len": round(float(part["text"].str.len().mean()), 1),
            "p99_len": int(np.percentile(part["text"].str.len(), 99)),
        }
    print("    ", len_stats)

    # ---------- 3. 字符级切分与词表 ----------
    print("[3/5] 构建字符级词表 ...")
    char_tokens = [char_tokenize(t) for t in train["text"]]
    char_vocab = build_vocab(char_tokens, min_freq=2, max_size=20000)
    save_json(char_vocab, os.path.join(args.outdir, "vocab_char.json"))
    print(f"    字符词表大小: {len(char_vocab)}")

    # ---------- 4. 词级切分（jieba，多进程）与词表 ----------
    print("[4/5] jieba 分词并构建词级词表 ...")
    import jieba
    jieba.setLogLevel(60)
    all_texts = df["text"].tolist()
    with Pool(args.workers) as pool:
        word_tokens_all = pool.map(word_tokenize, all_texts, chunksize=2000)
    n_tr = len(train)
    word_vocab = build_vocab(word_tokens_all[:n_tr], min_freq=2, max_size=50000)
    save_json(word_vocab, os.path.join(args.outdir, "vocab_word.json"))
    print(f"    词级词表大小: {len(word_vocab)}")

    # 保存词级切分结果（空格连接），供词级模型直接读取
    for name, part in zip(["train", "val", "test"],
                          [train, val, test]):
        idx = part.index.to_numpy()
        lines = [" ".join(word_tokens_all[i]) for i in idx]
        with open(os.path.join(args.outdir, f"{name}_word.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ---------- 5. 统计信息 ----------
    print("[5/5] 保存统计信息 ...")
    save_json({
        "n_raw": n_raw, "n_clean": n_clean, "n_spam": spam,
        "spam_ratio": round(spam / n_clean, 4),
        "char_max_len": args.char_max_len,
        "word_max_len": args.word_max_len,
        "vocab_char_size": len(char_vocab),
        "vocab_word_size": len(word_vocab),
        "split": len_stats,
    }, os.path.join(args.outdir, "stats.json"))
    print("完成。输出目录:", args.outdir)


if __name__ == "__main__":
    main()
