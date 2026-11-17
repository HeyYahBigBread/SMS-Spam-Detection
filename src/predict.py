# -*- coding: utf-8 -*-
"""推理演示：加载训练好的模型，对短信进行垃圾/正常分类。

用法：
    python src/predict.py --text "恭喜您中奖啦！点击链接领取奖金"
    python src/predict.py --file demo_sms.txt
    python src/predict.py --interactive
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import char_tokenize, load_json
from models import get_model


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    args = ckpt["args"]
    args["max_len"] = ckpt["max_len"]
    args["vocab_size"] = ckpt["vocab_size"]
    model = get_model(args["model"], ckpt["vocab_size"],
                      emb_dim=args["emb_dim"], hidden=args["hidden"],
                      num_layers=args["num_layers"], num_heads=args["num_heads"],
                      dim_ff=args["dim_ff"], dropout=args["dropout"],
                      pool=args.get("pool", "attn"))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, args


def predict(model, vocab, text, max_len, device):
    ids = [vocab.get(t, 1) for t in char_tokenize(text)[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="待判断的短信文本")
    ap.add_argument("--file", default=None, help="每行一条短信的文件")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--model-path",
                    default="results/models/bilstm_char_best.pt")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device != "auto"
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.model_path, device)
    vocab = load_json(os.path.join(args.data_dir,
                                   f"vocab_{cfg['level']}.json"))
    max_len = cfg["max_len"]

    def judge(text):
        p = predict(model, vocab, text, max_len, device)
        label = "垃圾短信" if p >= args.threshold else "正常短信"
        print(f"[{label}] (p={p:.4f}) {text}")

    if args.interactive:
        print("输入短信进行判断，输入 quit 退出：")
        while True:
            try:
                t = input("> ").strip()
            except EOFError:
                break
            if t.lower() in ("quit", "exit", "q"):
                break
            if t:
                judge(t)
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    judge(line)
    elif args.text:
        judge(args.text)
    else:
        print("请使用 --text / --file / --interactive 指定输入")


if __name__ == "__main__":
    main()
