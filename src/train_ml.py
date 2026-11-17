# -*- coding: utf-8 -*-
"""传统机器学习基线：TF-IDF + 朴素贝叶斯 / 逻辑回归 / 线性 SVM。

用法：
    python src/train_ml.py --level char --models nb,lr,svm
    python src/train_ml.py --level word --models nb,lr,svm
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import compute_metrics, save_json, set_seed

MODELS = {
    "nb": ("MultinomialNB", MultinomialNB()),
    "lr": ("LogisticRegression", LogisticRegression(max_iter=1000, C=1.0,
                                                    solver="liblinear")),
    "svm": ("LinearSVC", LinearSVC(C=1.0, max_iter=5000)),
}


def get_features(texts, level):
    """文本 -> TF-IDF 特征。"""
    if level == "char":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(1, 3), min_df=3,
            max_features=200000, sublinear_tf=True)
    else:
        vectorizer = TfidfVectorizer(
            token_pattern=r"\S+", min_df=3, max_features=200000,
            sublinear_tf=True)
    return vectorizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="char", choices=["char", "word"])
    ap.add_argument("--models", default="nb,lr,svm")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results/ml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/3] 加载数据 ({args.level} 级) ...")
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    if args.level == "word":
        for name, part in [("train", train), ("val", val), ("test", test)]:
            with open(os.path.join(args.data_dir, f"{name}_word.txt"),
                      encoding="utf-8") as f:
                part["text"] = [line.strip() for line in f
                                if line.strip()]
        assert len(train) == len(train["text"])

    vectorizer = get_features(train["text"].tolist(), args.level)
    t0 = time.time()
    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])
    X_test = vectorizer.transform(test["text"])
    print(f"    TF-IDF 特征维度: {X_train.shape[1]:,}，"
          f"向量化耗时 {time.time() - t0:.1f}s")

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    summary = {}
    for name in args.models.split(","):
        dname, clf = MODELS[name]
        print(f"[2/3] 训练 {dname} ...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        s_val = clf.decision_function(X_val)
        s_test = clf.decision_function(X_test)
        infer_time = time.time() - t0

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler().fit(np.concatenate([s_val, s_test])
                                    .reshape(-1, 1))
        p_val = clf.predict(X_val)
        p_test = clf.predict(X_test)
        score_test = scaler.transform(s_test.reshape(-1, 1)).ravel()

        m_val = compute_metrics(y_val, p_val, s_val)
        m_test = compute_metrics(y_test, p_test, score_test)
        tag = f"{args.level}_{name}"
        np.savez(os.path.join(args.out_dir, f"{tag}_test_pred.npz"),
                 y=y_test, pred=p_test, score=score_test)
        summary[tag] = {
            "model": dname, "level": args.level,
            "n_features": int(X_train.shape[1]),
            "fit_time_s": round(fit_time, 1),
            "infer_time_s": round(infer_time, 3),
            "val": m_val, "test": m_test,
        }
        print(f"    val:  acc={m_val['accuracy']:.4f} f1={m_val['f1']:.4f} "
              f"auc={m_val['auc']:.4f}")
        print(f"    test: acc={m_test['accuracy']:.4f} "
              f"pre={m_test['precision']:.4f} rec={m_test['recall']:.4f} "
              f"f1={m_test['f1']:.4f} auc={m_test['auc']:.4f} "
              f"(训练 {fit_time:.1f}s)")

    save_json(summary, os.path.join(args.out_dir, "summary.json"))
    print("[3/3] 完成，结果保存在", args.out_dir)


if __name__ == "__main__":
    main()
