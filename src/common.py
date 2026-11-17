# -*- coding: utf-8 -*-
"""公共工具：固定随机种子、数据读取、词汇表、数据集封装、评价指标。"""
import json
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_raw(path: str) -> pd.DataFrame:
    """读取原始数据。原始格式：每行第一个字符为标签(0/1)，其后为短信文本。"""
    labels, texts = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if len(line) < 2:
                continue
            lab, txt = line[0], line[1:].lstrip()
            if lab not in ("0", "1") or not txt:
                continue
            labels.append(int(lab))
            texts.append(txt)
    return pd.DataFrame({"label": labels, "text": texts})


_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """基础清洗：去 URL、去控制字符、压缩空白。"""
    text = _URL_RE.sub(" ", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def char_tokenize(text: str):
    """字符级切分（中文按字，英文/数字按连续片段）。"""
    return re.findall(r"[a-zA-Z0-9]+|[^\s]", text)


def word_tokenize(text: str):
    """词级切分（jieba 精确模式）。"""
    import jieba
    return [w.strip() for w in jieba.lcut(text) if w.strip()]


def build_vocab(token_lists, min_freq: int = 2, max_size: int = 20000):
    """根据 token 序列统计词频并建立词表（0=[PAD]，1=[UNK]）。"""
    from collections import Counter
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: -x[1])
    items = items[: max_size - 2]
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for w, _ in items:
        vocab[w] = len(vocab)
    return vocab


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SMSDataset(Dataset):
    """把 token 列表编码成定长 id 序列。"""

    def __init__(self, tokens_list, labels, vocab, max_len: int):
        self.max_len = max_len
        self.data = []
        for tokens, lab in zip(tokens_list, labels):
            ids = [vocab.get(t, 1) for t in tokens[:max_len]]
            ids = ids + [0] * (max_len - len(ids)) if len(ids) < max_len else ids
            self.data.append((ids, lab))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids, lab = self.data[i]
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(lab, dtype=torch.float32))


def compute_metrics(y_true, y_pred, y_score=None):
    """二分类评价指标（1 表示垃圾短信）。"""
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn_fp_fn_tp": confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist(),
    }
    if y_score is not None:
        try:
            res["auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            res["auc"] = None
    return res


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
