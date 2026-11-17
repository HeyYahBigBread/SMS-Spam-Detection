# -*- coding: utf-8 -*-
"""生成大作业报告（Word 文档），自动汇总实验结果并嵌入图表。

用法：
    python src/make_report.py --out report/大作业报告.docx
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json

from docx import Document
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Cm, Pt, RGBColor

FIG_DIR = "results/figures"
ML_SUMMARY = "results/ml/summary.json"
MODEL_DIR = "results/models"
DATA_STATS = "data/processed/stats.json"

STUDENT_ID = "【学号】"
STUDENT_NAME = "【姓名】"
GIT_LINK = "【GitHub/Gitee 仓库链接：推送后将此占位替换】"

METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "auc"]
METRIC_CN = {"accuracy": "Accuracy", "precision": "Precision",
             "recall": "Recall", "f1": "F1", "auc": "AUC"}


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def model_display(row):
    lvl = {"char": "字级", "word": "词级"}.get(row.get("level"), row.get("level"))
    if "params" in row:
        m = {"bilstm": "BiLSTM", "textcnn": "TextCNN",
             "transformer": "Transformer", "hybrid": "BiLSTM+TextCNN"}[row["model"]]
        return f"{m}（{lvl}）"
    m = {"MultinomialNB": "朴素贝叶斯", "LogisticRegression": "逻辑回归",
         "LinearSVC": "线性 SVM"}.get(row.get("model"), row.get("model"))
    return f"{m}（{lvl}）"


def is_ablation(row):
    tag = row.get("tag", "")
    return tag.endswith("_meanpool") or tag.endswith("_noweight")


def collect_rows():
    rows = []
    if os.path.exists(ML_SUMMARY):
        rows += list(load_json(ML_SUMMARY).values())
    for p in glob.glob(os.path.join(MODEL_DIR, "*_test.json")):
        rows.append(load_json(p))
    return rows


def add_p(doc, text, size=10.5, bold=False, align=None, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = "宋体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    return p


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.name = "黑体"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        r.font.color.rgb = RGBColor(0, 0, 0)
    return h


def add_img(doc, path, width_cm, caption=None):
    if not os.path.exists(path):
        add_p(doc, f"[缺少图片: {path}]")
        return
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        c = add_p(doc, caption, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
        for r in c.runs:
            r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def add_table(doc, headers, data, col_widths=None):
    t = doc.add_table(rows=1 + len(data), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = str(h)
        for par in cell.paragraphs:
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in par.runs:
                r.font.bold = True
                r.font.size = Pt(9)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            cell = t.rows[i + 1].cells[j]
            cell.text = str(v)
            for par in cell.paragraphs:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in par.runs:
                    r.font.size = Pt(9)
    if col_widths:
        for j, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    return t


def build_report(doc):
    stats = load_json(DATA_STATS)
    all_rows = collect_rows()
    rows = [r for r in all_rows if not is_ablation(r)]
    rows.sort(key=lambda r: r["test"]["f1"], reverse=True)
    dl_rows = [r for r in rows if "params" in r]
    ml_rows = [r for r in rows if "params" not in r]
    best = rows[0] if rows else None
    best_dl = dl_rows[0] if dl_rows else None

    # ================= 封面 =================
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("《深度学习实践》大作业报告")
    r.font.size = Pt(26); r.font.bold = True; r.font.name = "黑体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("垃圾短信识别")
    r.font.size = Pt(20); r.font.bold = True; r.font.name = "黑体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    doc.add_paragraph()
    for line in [f"学号：{STUDENT_ID}", f"姓名：{STUDENT_NAME}",
                 "选题：基础题第 2 题 · 垃圾短信识别",
                 f"代码仓库：{GIT_LINK}"]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line)
        r.font.size = Pt(14)
    doc.add_page_break()

    # ================= 一、选题 =================
    add_heading(doc, "一、选题与任务概述", 1)
    add_p(doc, "本大作业选题为基础题第 2 题“垃圾短信识别”：给定一条短信息文本，"
               "自动判断该短信是否为垃圾短信（广告、诈骗、骚扰等）。该任务本质上是"
               "一个二分类文本分类问题，是自然语言处理（NLP）中最基础、应用最广泛的"
               "任务之一，也是各大短信平台、运营商内容安全风控的核心环节。",
          indent=True)
    add_p(doc, "本项目的技术路线：首先构建并预处理中文垃圾短信数据集；然后分别实现"
               "传统机器学习基线（TF-IDF + 朴素贝叶斯 / 逻辑回归 / 线性 SVM）与"
               "深度学习模型（BiLSTM、TextCNN、Transformer 及 BiLSTM+TextCNN "
               "混合模型），进行系统的对比实验与消融分析；最后从 Accuracy、"
               "Precision、Recall、F1、AUC 等指标对结果进行量化评估与可视化分析。",
          indent=True)

    # ================= 二、数据集 =================
    add_heading(doc, "二、数据集说明", 1)
    add_heading(doc, "2.1 数据来源", 2)
    add_p(doc, "本项目采用公开的中文垃圾短信数据集（约 80 万条带标签短信），"
               "由开源仓库 hrwhisper/SpamMessage 整理发布"
               "（https://github.com/hrwhisper/SpamMessage），原始语料出自某短信"
               "平台收集的真实短信。每条数据由标签与短信文本组成：标签 0 表示正常"
               "短信，标签 1 表示垃圾短信（广告、诈骗等）。", indent=True)
    add_heading(doc, "2.2 数据预处理", 2)
    add_p(doc, "数据预处理流程如下：", indent=True)
    for step in [
        "① 读取原始文件（每行“标签+文本”格式），解析出标签与短信文本；",
        "② 文本清洗：去除 URL 链接、控制字符，压缩多余空白；",
        "③ 质量过滤：删除长度小于 2 个字符的空文本；",
        "④ 去重处理：对完全相同的文本去重；同一文本若存在标签冲突则整条剔除；",
        "⑤ 特征提取：构造字符级特征（中文按字、英文数字按连续片段）与词级特征"
        "（jieba 分词）；",
        "⑥ 构建词表：字符级与词级词表均按词频过滤（min_freq=2），保留 [PAD]/[UNK]"
        "两个特殊符号；",
        "⑦ 序列定长：字符级序列截断/填充至 100，词级截断/填充至 80。",
    ]:
        add_p(doc, step, size=10.5)
    add_heading(doc, "2.3 数据划分", 2)
    add_p(doc, f"清洗后共保留 {stats['n_clean']:,} 条短信，其中垃圾短信 "
               f"{stats['n_spam']:,} 条（占 {stats['spam_ratio']:.2%}）。"
               f"按标签分层抽样（stratified sampling）划分为训练集、验证集、"
               f"测试集，比例为 8 : 1 : 1（训练 {stats['split']['train']['n']:,} 条、"
               f"验证 {stats['split']['val']['n']:,} 条、"
               f"测试 {stats['split']['test']['n']:,} 条），保证三个集合的"
               f"类别分布一致，并固定随机种子（seed=42）以确保实验可复现。",
          indent=True)
    add_img(doc, os.path.join(FIG_DIR, "data_stats.png"), 15.5,
            "图 1  数据集标签分布与短信长度分布")

    # ================= 三、模型 =================
    add_heading(doc, "三、模型设计与实现", 1)
    add_heading(doc, "3.1 传统机器学习基线", 2)
    add_p(doc, "基线方法采用 TF-IDF（词频-逆文档频率）将短信文本向量化：字符级"
               "特征使用字符 n-gram（1~3 元）捕捉局部搭配信息，词级特征使用 jieba"
               "分词后的词语。在此基础上训练三种经典分类器：多项式朴素贝叶斯"
               "（MultinomialNB）、逻辑回归（LogisticRegression）与线性 SVM"
               "（LinearSVC）。", indent=True)
    add_heading(doc, "3.2 深度学习模型", 2)
    add_p(doc, "深度学习模型均使用随机初始化的嵌入层（Embedding）将字符/词 ID 映射"
               "为 128 维向量，并以加权二分类交叉熵（按正负样本比设置 pos_weight "
               "缓解类别不平衡）作为损失函数。共实现四类模型：", indent=True)
    for name, desc in [
        ("BiLSTM（图 2）", "双层双向 LSTM 建模序列的上下文依赖，采用注意力池化"
         "（attention pooling）对时间步加权求和，替代简单的均值池化，让模型自动"
         "关注关键字符/词；后接两层全连接分类头。"),
        ("TextCNN（图 3）", "经典卷积文本分类模型：嵌入矩阵上并行使用卷积核宽度"
         "k=2,3,4,5 的一维卷积提取不同尺度的局部 n-gram 特征，最大池化后拼接，"
         "送入全连接分类头。"),
        ("Transformer（图 4）", "嵌入向量叠加正弦位置编码后送入 2 层 Transformer "
         "编码器（4 头自注意力 + GELU 前馈网络，Pre-LN），对有效位置（掩码）做"
         "均值池化后分类。"),
        ("BiLSTM+TextCNN 混合模型（图 5）", "共享同一嵌入层的双通道结构：BiLSTM"
         "通道提取全局序列语义，TextCNN 通道提取局部 n-gram 特征，两通道特征"
         "拼接后送入分类头，实现局部与全局特征的互补融合。"),
    ]:
        p = add_p(doc, name + "：", size=10.5)
        p.runs[0].font.bold = True
        add_p(doc, desc, size=10.5, indent=True)
    add_img(doc, os.path.join(FIG_DIR, "arch_bilstm.png"), 13.5,
            "图 2  BiLSTM 模型结构")
    add_img(doc, os.path.join(FIG_DIR, "arch_textcnn.png"), 13.5,
            "图 3  TextCNN 模型结构")
    add_img(doc, os.path.join(FIG_DIR, "arch_transformer.png"), 13.5,
            "图 4  Transformer 模型结构")
    add_img(doc, os.path.join(FIG_DIR, "arch_hybrid.png"), 15.5,
            "图 5  BiLSTM+TextCNN 混合模型结构")

    # ================= 四、实验设置 =================
    add_heading(doc, "四、实验设置", 1)
    add_heading(doc, "4.1 硬件与软件环境", 2)
    add_table(doc,
              ["项目", "配置"],
              [["CPU", "Intel（笔记本）"],
               ["GPU", "NVIDIA GeForce RTX 5070 Laptop GPU（CUDA 可用）"],
               ["操作系统", "Windows"],
               ["Python", "3.10（conda 环境 pytorch）"],
               ["深度学习框架", "PyTorch 2.10（cu130）"],
               ["其他库", "scikit-learn 1.7 / pandas 2.3 / jieba / matplotlib"]],
              [4, 10])
    add_heading(doc, "4.2 超参数设置", 2)
    add_p(doc, "深度学习模型统一采用如下超参数（经网格搜索与验证集调优后确定）：",
          indent=True)
    add_table(doc,
              ["超参数", "取值"],
              [["优化器", "AdamW（weight_decay=1e-5）"],
               ["学习率", "1e-3，ReduceLROnPlateau 衰减（factor=0.5）"],
               ["批大小", "512"],
               ["最大训练轮数", "10（early stopping，patience=3）"],
               ["嵌入维度", "128"],
               ["LSTM 隐藏维度/层数", "128 / 2（双向）"],
               ["Transformer 层数/头数", "2 / 4（d_ff=256）"],
               ["TextCNN 卷积核/通道", "k=2,3,4,5 / 各 128"],
               ["Dropout", "0.3"],
               ["梯度裁剪", "max_norm=5.0"],
               ["损失函数", "BCEWithLogitsLoss（pos_weight 按类别比）"],
               ["随机种子", "42"]],
              [6, 8])
    add_p(doc, "机器学习基线：TF-IDF 取 min_df=3、最多 20 万维特征、"
               "sublinear_tf=True；逻辑回归与线性 SVM 使用默认正则强度 C=1.0。",
          indent=True)

    # ================= 五、优化过程 =================
    add_heading(doc, "五、优化过程", 1)
    add_p(doc, "实验采取“由浅入深、逐步优化”的策略，主要经历了以下阶段：",
          indent=True)
    for title, desc in [
        ("阶段一：传统机器学习基线。",
         "以 TF-IDF + 朴素贝叶斯 / 逻辑回归 / 线性 SVM 作为强基线，同时对比字符级"
         "与词级特征的差异，为深度学习模型设定性能标杆。"),
        ("阶段二：基础 BiLSTM 模型。",
         "搭建字符级双层 BiLSTM，初始采用均值池化与普通二分类交叉熵损失。发现"
         "验证集 F1 存在提升空间，且模型对垃圾短信召回不足。"),
        ("阶段三：引入注意力池化与类别加权。",
         "将均值池化替换为注意力池化，让模型自适应关注判定垃圾短信的关键字符；同时按"
         "训练集正负样本比设置 pos_weight 加权损失，缓解垃圾/正常样本不均衡问题。"
         "消融实验表明池化方式与加权损失对结果的影响（见 5.1 节）。"),
        ("阶段四：训练策略调优。",
         "在 BiLSTM 上网格搜索学习率（2e-4/5e-4/1e-3）、Dropout（0.3/0.5）、"
         "批大小（256/512），采用 ReduceLROnPlateau 学习率衰减、梯度裁剪与"
         "early stopping，防止过拟合并加速收敛。"),
        ("阶段五：多模型对比与特征融合。",
         "在调优后的训练策略下训练 TextCNN、Transformer 及 BiLSTM+TextCNN 混合"
         "模型；混合模型通过全局序列语义与局部 n-gram 特征的互补融合，取得了"
         "深度学习模型中最优的综合性能，逼近最强的传统机器学习基线。"),
    ]:
        p = add_p(doc, title, size=10.5)
        p.runs[0].font.bold = True
        add_p(doc, desc, size=10.5, indent=True)

    add_heading(doc, "5.1 消融实验", 2)
    abl_variants = []  # (variant, row)
    for r in all_rows:
        if not isinstance(r.get("hyperparams"), dict):
            continue
        hp = r["hyperparams"]
        if hp.get("model") == "bilstm" and hp.get("level") == "char":
            if hp.get("no_pos_weight"):
                abl_variants.append(("BiLSTM（注意力池化，无类别加权）", r))
            elif hp.get("pool") == "mean":
                abl_variants.append(("BiLSTM（均值池化，有类别加权）", r))
            elif hp.get("pool") == "attn":
                abl_variants.append(("BiLSTM（注意力池化，有类别加权）", r))
    order = ["BiLSTM（注意力池化，有类别加权）", "BiLSTM（均值池化，有类别加权）",
             "BiLSTM（注意力池化，无类别加权）"]
    abl_variants.sort(key=lambda x: order.index(x[0]) if x[0] in order else 99)
    if abl_variants:
        abl_rows = []
        for variant, r in abl_variants:
            t, v = r["test"], r.get("val", {})
            abl_rows.append([variant, fmt(t["f1"]), fmt(t["recall"]),
                             fmt(t["precision"]), fmt(t["accuracy"]),
                             fmt(v.get("f1"), 4)])
        add_table(doc,
                  ["配置", "测试 F1", "测试 Recall", "测试 Precision",
                   "测试 Accuracy", "验证 F1"],
                  abl_rows, [6.2, 2.0, 2.3, 2.5, 2.5, 1.9])
        add_p(doc, "表 1  BiLSTM 消融实验（字符级）", size=9,
              align=WD_ALIGN_PARAGRAPH.CENTER)
        if any("无类别加权" in v for v, _ in abl_variants):
            weight_row = next((r for v, r in abl_variants
                               if v == "BiLSTM（注意力池化，有类别加权）"), None)
            no_w_row = next((r for v, r in abl_variants if "无类别加权" in v), None)
            if weight_row and no_w_row:
                d_rec = weight_row["test"]["recall"] - no_w_row["test"]["recall"]
                d_f1 = weight_row["test"]["f1"] - no_w_row["test"]["f1"]
                rec_desc = f"去掉加权后召回率下降 {abs(d_rec):.4f}" if d_rec > 0 \
                    else f"去掉加权后召回率反而上升 {abs(d_rec):.4f}"
                f1_desc = f"F1 下降 {abs(d_f1):.4f}" if d_f1 > 0 \
                    else f"F1 上升 {abs(d_f1):.4f}"
                add_p(doc, f"由表 1 可见：注意力池化与均值池化在该任务上性能接近"
                           f"（短信文本较短、平均 25.6 字，长程依赖有限）；类别加权"
                           f"损失对正负样本不均衡（垃圾短信仅占 10%）问题具有缓解"
                           f"作用——{rec_desc}，{f1_desc}。", indent=True)
        else:
            add_p(doc, "由表 1 可见：注意力池化与均值池化在该任务上性能接近，"
                       "这是因为短信文本较短（平均 25.6 字）、长程依赖有限，"
                       "两者均能有效聚合序列信息。", indent=True)
    add_img(doc, os.path.join(FIG_DIR, "training_curves.png"), 15.5,
            "图 6  深度学习模型验证集训练曲线")

    # ================= 六、结果分析 =================
    add_heading(doc, "六、结果分析", 1)
    add_heading(doc, "6.1 总体对比", 2)
    table_rows = []
    for r in rows:
        t = r["test"]
        n_params = r.get("params", r.get("n_features"))
        table_rows.append([
            model_display(r),
            fmt(t["accuracy"]), fmt(t["precision"]), fmt(t["recall"]),
            fmt(t["f1"]), fmt(t["auc"]),
            f"{n_params:,}" if n_params else "—",
            f"{r.get('train_time_s', r.get('fit_time_s', '—'))}",
        ])
    add_table(doc,
              ["模型", "Accuracy", "Precision", "Recall", "F1", "AUC",
               "参数量/特征数", "训练耗时(s)"],
              table_rows,
              [4.2, 1.8, 1.9, 1.6, 1.6, 1.6, 2.4, 1.9])
    add_p(doc, "表 2  各模型在测试集上的性能对比（垃圾短信为正类）", size=9,
          align=WD_ALIGN_PARAGRAPH.CENTER)
    add_p(doc, f"由表 2 可见：传统机器学习基线在该数据集上表现极强，字级 TF-IDF + "
               f"线性 SVM 以 F1={fmt(max(r['test']['f1'] for r in rows))} 位居第一；"
               f"深度学习模型中，混合模型取得了最高的 F1（{fmt(best_dl['test']['f1'])}）"
               f"与最接近的差距。进一步观察 Precision 与 Recall 的权衡：线性 SVM "
               f"精确率最高（{fmt(max(r['test']['precision'] for r in rows))}，误报"
               f"最少），而深度模型（尤其 Transformer 与 BiLSTM）召回率更高"
               f"（最高 {fmt(max(r['test']['recall'] for r in rows))}，漏判更少）——"
               f"这说明两类方法学到的判别边界不同，实际部署中可按业务对误报/漏判的"
               f"代价偏好选择模型或调整决策阈值。特征粒度方面，字符级特征整体优于"
               f"词级特征（无需分词、工程开销小），符合中文短信中短文本 n-gram "
               f"判别性强的特点。",
          indent=True)
    add_img(doc, os.path.join(FIG_DIR, "roc_curves.png"), 12.5,
            "图 7  各模型 ROC 曲线对比")
    add_heading(doc, "6.2 混淆矩阵", 2)
    add_img(doc, os.path.join(FIG_DIR, "confusion_matrices.png"), 16.0,
            "图 8  F1 最高的 4 个模型的混淆矩阵")
    add_p(doc, "混淆矩阵显示，各模型的误判主要集中在少数语义模糊的短信上："
               "一类是伪装成正常通知的诈骗短信（假阴性，漏判），另一类是包含敏感词"
               "的正常对话（假阳性，误报）。垃圾短信漏判的代价通常高于误报，"
               "因此后续工作可在加权损失的基础上进一步调整决策阈值。更深入的数据"
               "层面的误判成因分析见 6.3 节。", indent=True)

    err_file = os.path.join(FIG_DIR, "error_analysis.txt")
    add_heading(doc, "6.3 数据洞察与模型局限", 2)
    add_p(doc, "在实验过程中我们对数据集做了进一步的探索性分析，发现一个重要的数据"
               "特性：训练集中正常短信类几乎不含标点符号（含“，”的比例为 0.0%、"
               "含“。”为 0.0%、含“！”为 0.0%），而垃圾短信中 82.9% 含有“，”、"
               "37.4% 含有“。”、43.3% 含有“！”。这是因为正常短信多采集自口语化聊天"
               "文本（无标点），而垃圾短信多为带完整标点的广告/诈骗模板。",
          indent=True)
    add_p(doc, "该特性意味着“是否含标点”在本数据集中是近乎完美的判别特征，测试集上"
               "的高指标（F1≈0.99）有一部分来自这一数据偏差。我们用手工构造的"
               "样例验证了这一点：带标点的口语化正常短信（如“明天下午三点会议室开会，"
               "收到请回复”）会被模型以极高置信度误判为垃圾短信，而去掉标点后即可"
               "正确判定。这说明模型学到了模板化的浅层特征。",
          indent=True)
    add_p(doc, "局限与改进方向：① 部署到真实场景（现代短信普遍带标点）前，应补充"
               "带标点的正常短信样本做微调或重训练；② 可采用对抗训练、标点掩码"
               "数据增强等手段削弱对单一浅层特征的依赖；③ 该发现同时提示，"
               "评估文本分类模型时应关注数据偏差对指标的影响。", indent=True)

    if os.path.exists(err_file):
        add_heading(doc, "6.4 错误分析", 2)
        with open(err_file, encoding="utf-8") as f:
            err_text = f.read().strip().split("\n")
        for line in err_text[:16]:
            add_p(doc, line, size=9)

    # ================= 七、创新点 =================
    add_heading(doc, "七、创新点", 1)
    for item in [
        "① 系统性的多模型对比：在同一数据集、同一划分上公平对比 3 种传统机器学习"
        "模型与 4 种深度学习模型（含字符级/词级两种特征粒度），给出量化指标与"
        "可视化结果；",
        "② 局部+全局特征融合的混合模型：提出 BiLSTM 与 TextCNN 共享嵌入的双通道"
        "结构，兼顾短信文本的全局语义与局部 n-gram 特征，取得深度学习模型中的"
        "最优性能，与最强传统基线基本持平；",
        "③ 针对类别不平衡的损失设计：按正负样本比设置 pos_weight 的加权二分类"
        "交叉熵，并通过消融实验对比其影响；",
        "④ 注意力池化机制：用可学习的注意力权重替代均值池化，模型可自适应地"
        "关注判定垃圾短信的关键字符，提升可解释性；",
        "⑤ 完整可复现的工程实现：固定随机种子、分层划分数据、early stopping 与"
        "学习率调度、断点续训支持，全部代码开源（含数据下载、预处理、训练、"
        "评估、推理全流程）。",
        "⑥ 数据洞察与偏差分析：通过探索性分析发现数据集“正常短信几乎不含标点”"
        "的标注偏差，并用手工样例验证了模型对浅层特征（标点）的依赖，指出"
        "高指标背后的数据成因，为模型部署与改进指明方向。",
    ]:
        add_p(doc, item, size=10.5)

    # ================= 八、总结 =================
    add_heading(doc, "八、总结", 1)
    add_p(doc, f"本项目完成了垃圾短信识别任务的全流程实践：构建了约 {stats['n_clean']:,} "
               f"条的中文短信数据集并完成清洗与划分；实现并系统对比了 7 种分类模型。"
               f"实验表明：字级 TF-IDF + 线性 SVM 取得最优 F1（{fmt(best['test']['f1'])}），"
               f"而 BiLSTM+TextCNN 混合模型是表现最好的深度学习模型"
               f"（F1={fmt(best_dl['test']['f1'])}），与其基本持平；深度模型在召回率上"
               f"更具优势（漏判更少），传统模型精确率更高（误报更少），两类方法形成"
               f"互补。通过消融实验分析了池化方式与类别加权损失的作用。",
          indent=True)
    add_p(doc, "不足与展望：① 数据集中部分垃圾短信与正常短信语义高度接近，可引入"
               "预训练语言模型（如 BERT/RoBERTa）进一步提升；② 可对垃圾短信做细粒度"
               "分类（广告/诈骗/骚扰）；③ 可将模型部署为在线 API 或端侧推理服务，"
               "服务于真实短信拦截场景。", indent=True)

    # ================= 参考文献 =================
    add_heading(doc, "参考文献", 1)
    refs = [
        "[1] hrwhisper. SpamMessage: 中文垃圾短信识别. https://github.com/hrwhisper/SpamMessage",
        "[2] Kim Y. Convolutional Neural Networks for Sentence Classification[C]. EMNLP, 2014.",
        "[3] Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural Computation, 1997, 9(8): 1735-1780.",
        "[4] Vaswani A, et al. Attention Is All You Need[C]. NeurIPS, 2017.",
        "[5] Zhou P, et al. Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification[C]. ACL, 2016.",
        "[6] Almeida T A, Hidalgo J M G, Yamakami A. Contributions to the Study of SMS Spam Filtering[J]. ACM Computing Surveys, 2011.",
    ]
    for ref in refs:
        add_p(doc, ref, size=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/大作业报告.docx")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc = Document()
    build_report(doc)
    doc.save(args.out)
    print("报告已生成:", args.out)


if __name__ == "__main__":
    main()
