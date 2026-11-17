# -*- coding: utf-8 -*-
"""深度学习模型训练脚本（BiLSTM / TextCNN / Transformer / Hybrid，字级或词级）。

用法：
    python src/train_dl.py --model bilstm --level char \
        --epochs 10 --batch-size 512 --lr 1e-3
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, load_json, char_tokenize, word_tokenize,
                    SMSDataset, compute_metrics, get_device, save_json)
from models import get_model, count_params


def load_tokens(split, level, data_dir):
    """读取某个划分的 token 列表。"""
    if level == "char":
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        return [char_tokenize(t) for t in df["text"]], \
            df["label"].tolist()
    else:
        df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        with open(os.path.join(data_dir, f"{split}_word.txt"),
                  encoding="utf-8") as f:
            tokens = [line.strip().split() for line in f if line.strip()]
        return tokens, df["label"].tolist()


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    preds, scores, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logit = model(x)
            prob = torch.sigmoid(logit)
            preds.append((prob > threshold).long().cpu())
            scores.append(prob.cpu())
            labels.append(y.long().cpu())
    preds = torch.cat(preds).numpy()
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    return labels, preds, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bilstm",
                    choices=["bilstm", "textcnn", "transformer", "hybrid"])
    ap.add_argument("--level", default="char", choices=["char", "word"])
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results/models")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dim-ff", type=int, default=256)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--pool", default="attn", choices=["attn", "mean"],
                    help="BiLSTM 池化方式（消融实验用）")
    ap.add_argument("--no-pos-weight", action="store_true",
                    help="关闭类别不平衡加权（消融实验用）")
    ap.add_argument("--resume", action="store_true",
                    help="从已有 checkpoint 断点续训")
    ap.add_argument("--tag-suffix", default="",
                    help="结果文件 tag 后缀（消融实验用，如 meanpool）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"设备: {device}")

    vocab = load_json(os.path.join(args.data_dir,
                                   f"vocab_{args.level}.json"))
    max_len = 100 if args.level == "char" else 80
    if args.level == "word" and len(vocab) < args.hidden:
        print("[警告] 词级词表较小，嵌入层可能欠拟合")

    print(f"[1/4] 加载数据 ({args.level} 级, max_len={max_len}) ...")
    sets = {}
    for split in ["train", "val", "test"]:
        tokens, labels = load_tokens(split, args.level, args.data_dir)
        sets[split] = SMSDataset(tokens, labels, vocab, max_len)
        print(f"    {split}: {len(sets[split])} 条")

    train_loader = torch.utils.data.DataLoader(
        sets["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        sets["val"], batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        sets["test"], batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True)

    model = get_model(
        args.model, len(vocab), emb_dim=args.emb_dim, hidden=args.hidden,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dim_ff=args.dim_ff, dropout=args.dropout, pool=args.pool).to(device)
    n_params = count_params(model)
    print(f"[2/4] 模型 {args.model} 参数量: {n_params:,}")

    # 类别不平衡：按训练集正负样本比设置 pos_weight（消融实验可关闭）
    train_labels = sets["train"].data
    n_pos = sum(lab for _, lab in train_labels)
    n_neg = len(train_labels) - n_pos
    if args.no_pos_weight:
        pos_weight = torch.tensor([1.0], device=device)
        print("    已关闭 pos_weight（消融实验）")
    else:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        print(f"    正样本 {n_pos:,} / 负样本 {n_neg:,} -> pos_weight={n_neg / n_pos:.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1)

    tag = f"{args.model}_{args.level}"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    os.makedirs(args.out_dir, exist_ok=True)
    hist_path = os.path.join(args.out_dir, f"{tag}_history.csv")
    ckpt_path = os.path.join(args.out_dir, f"{tag}_best.pt")
    progress_path = os.path.join(args.out_dir, f"{tag}_progress.json")

    # ---------- 断点续训 ----------
    start_epoch, best_val_f1, best_epoch, bad_epochs = 1, 0.0, 0, 0
    history = []
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if os.path.exists(progress_path):
            prog = load_json(progress_path)
            start_epoch = prog["epoch"] + 1
            best_val_f1 = prog["best_val_f1"]
            best_epoch = prog["best_epoch"]
        if os.path.exists(hist_path):
            with open(hist_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                history = list(reader)
        print(f"    断点续训：从第 {start_epoch} 轮继续 "
              f"(best_val_f1={best_val_f1:.4f} @ epoch {best_epoch})")

    print(f"[3/4] 开始训练（第 {start_epoch}~{args.epochs} 轮）...")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tot_loss, n_batch = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot_loss += loss.item()
            n_batch += 1
        tr_loss = tot_loss / n_batch

        yv, pv, sv = evaluate(model, val_loader, device)
        m_val = compute_metrics(yv, pv, sv)
        scheduler.step(tr_loss)

        history.append({
            "epoch": epoch, "train_loss": round(tr_loss, 4),
            "val_loss": None, "val_acc": m_val["accuracy"],
            "val_f1": m_val["f1"], "val_auc": m_val["auc"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        flag = ""
        if m_val["f1"] > best_val_f1:
            best_val_f1, best_epoch, bad_epochs = m_val["f1"], epoch, 0
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args), "vocab_size": len(vocab),
                        "max_len": max_len},
                       ckpt_path)
            save_json({"epoch": epoch, "best_val_f1": best_val_f1,
                       "best_epoch": best_epoch}, progress_path)
            flag = " *"
        else:
            bad_epochs += 1
        print(f"  epoch {epoch:2d} | train_loss {tr_loss:.4f} | "
              f"val_acc {m_val['accuracy']:.4f} | val_f1 {m_val['f1']:.4f} | "
              f"val_auc {m_val['auc']:.4f}{flag}")
        if bad_epochs >= args.patience:
            print(f"  early stopping（连续 {args.patience} 轮无提升）")
            break
    train_time = time.time() - t0
    print(f"    训练耗时 {train_time:.1f}s，最佳轮次 {best_epoch}")

    # 写入训练历史
    if history:
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    # ---------- 测试集评估 ----------
    print("[4/4] 测试集评估 ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    yt, pt, st = evaluate(model, test_loader, device)
    m_test = compute_metrics(yt, pt, st)

    np.savez(os.path.join(args.out_dir, f"{tag}_test_pred.npz"),
             y=yt, pred=pt, score=st)
    save_json({
        "tag": tag,
        "model": args.model, "level": args.level,
        "params": n_params,
        "best_epoch": best_epoch,
        "train_time_s": round(train_time, 1),
        "test": m_test,
        "hyperparams": {k: v for k, v in vars(args).items()
                        if k not in ("data_dir", "out_dir")},
    }, os.path.join(args.out_dir, f"{tag}_test.json"))
    print("    测试集指标:", {k: m_test[k] for k in
                            ("accuracy", "precision", "recall", "f1", "auc")})


if __name__ == "__main__":
    main()
