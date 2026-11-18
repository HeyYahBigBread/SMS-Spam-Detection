# 垃圾短信识别（SMS Spam Detection）

《深度学习实践》课程大作业 · 基础题第 2 题：识别短信是否为广告、诈骗等垃圾短信。

本项目对比了 **传统机器学习**（TF-IDF + 朴素贝叶斯 / 逻辑回归 / 线性 SVM）与
**深度学习**（BiLSTM、TextCNN、Transformer、BiLSTM+TextCNN 混合模型）在
中文垃圾短信识别任务上的表现。

## 环境

- Python 3.10（conda 环境 `pytorch`）
- PyTorch 2.x（CUDA 可用，本项目使用 RTX 5070 Laptop GPU 训练）
- 依赖安装：`pip install -r requirements.txt`

## 数据集

**来源**：中文垃圾短信数据集（约 80 万条带标签短信），由
[hrwhisper/SpamMessage](https://github.com/hrwhisper/SpamMessage) 公开提供
（原始数据出自某互联网短信语料库，标签 0=正常短信，1=垃圾短信）。

**下载**（任选其一）：

```bash
# 直接下载
curl -L -o data/raw/labeled_sms.txt \
  "https://gh-proxy.com/https://raw.githubusercontent.com/hrwhisper/SpamMessage/master/data/%E5%B8%A6%E6%A0%87%E7%AD%BE%E7%9F%AD%E4%BF%A1.txt"

# 或从 GitHub 仓库获取 data/带标签短信.txt（58MB）
```

原始格式：每行 = 标签（首字符 0/1） + 短信文本。

**预处理**（`src/preprocess.py`）：

1. 清洗：去除 URL、控制字符，压缩空白；
2. 过滤：删除长度 < 2 的文本；同一文本若标签冲突则剔除，其余去重；
3. 划分：按标签分层抽样，train : val : test = 8 : 1 : 1（固定随机种子 42）；
4. 词表：字符级词表（min_freq=2）与 jieba 词级词表（min_freq=2）；
5. 序列长度：字符级截断/填充到 100，词级 80。

## 模型

| 类型 | 模型 | 说明 |
| --- | --- | --- |
| 传统 ML | 朴素贝叶斯 / 逻辑回归 / 线性 SVM | TF-IDF 特征（字符 n-gram 1~3 或 jieba 分词） |
| 深度学习 | BiLSTM | 双层双向 LSTM + 注意力池化 |
| 深度学习 | TextCNN | 多卷积核 (2,3,4,5) + 最大池化 |
| 深度学习 | Transformer | 位置编码 + 2 层 TransformerEncoder + 掩码均值池化 |
| 深度学习 | Hybrid | BiLSTM 与 TextCNN 双通道特征融合 |

## 运行

```bash
# 1. 预处理（清洗、划分、构建词表）
python src/preprocess.py

# 2. 传统机器学习基线
python src/train_ml.py --level char --models nb,lr,svm
python src/train_ml.py --level word --models nb,lr,svm

# 3. 深度学习模型（可选 --model bilstm/textcnn/transformer/hybrid）
python src/train_dl.py --model bilstm --level char --epochs 10 --batch-size 512 --lr 1e-3
python src/train_dl.py --model textcnn --level char
python src/train_dl.py --model transformer --level char
python src/train_dl.py --model hybrid --level char

# 4. 汇总评估与可视化（对比表、ROC、混淆矩阵、训练曲线）
python src/evaluate.py

# 5. 模型结构图
python src/plot_arch.py

# 6. 推理演示
python src/predict.py --text "恭喜您中奖啦！点击链接领取万元奖金"
python src/predict.py --interactive
```

训练与评估的输出（模型权重、指标 JSON、图表）保存在 `results/` 目录
（该目录不入库，可由上述命令随时重新生成）。

## 目录结构

```
├── data/                    # 数据集（不入库，见上方下载说明）
├── src/
│   ├── common.py            # 工具函数（词表、数据集、指标）
│   ├── preprocess.py        # 数据预处理
│   ├── models.py            # 深度学习模型定义
│   ├── train_ml.py          # 机器学习基线训练
│   ├── train_dl.py          # 深度学习模型训练（支持断点续训）
│   ├── evaluate.py          # 统一评估与可视化
│   ├── plot_arch.py         # 模型结构图
│   ├── plot_data_stats.py   # 数据集统计图
│   └── predict.py           # 推理演示
├── demo_sms.txt             # 推理演示样例（前 5 条垃圾 / 后 5 条正常）
├── requirements.txt
└── README.md
```

## 结果概览

测试集（78,646 条）上各模型对比（运行 `python src/evaluate.py` 可复现全部
指标与图表）：

| 模型 | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| 线性 SVM（字级 TF-IDF） | 0.9984 | 0.9980 | 0.9857 | **0.9918** | 0.9997 |
| BiLSTM+TextCNN 混合（字级） | 0.9982 | 0.9930 | 0.9889 | 0.9909 | 0.9995 |
| TextCNN（字级） | 0.9980 | 0.9931 | 0.9874 | 0.9902 | 0.9995 |
| 线性 SVM（词级 TF-IDF） | 0.9979 | 0.9977 | 0.9815 | 0.9895 | 0.9992 |
| BiLSTM（字级） | 0.9977 | 0.9881 | 0.9894 | 0.9888 | 0.9996 |
| Transformer（字级） | 0.9974 | 0.9833 | 0.9905 | 0.9869 | 0.9997 |
| 逻辑回归（字级） | 0.9971 | 0.9963 | 0.9748 | 0.9854 | 0.9995 |
| 逻辑回归（词级） | 0.9968 | 0.9979 | 0.9703 | 0.9839 | 0.9995 |
| 朴素贝叶斯（词级） | 0.9950 | 0.9656 | 0.9856 | 0.9755 | 0.9994 |
| 朴素贝叶斯（字级） | 0.9923 | 0.9378 | 0.9895 | 0.9630 | 0.9995 |

结论：传统方法（字级 n-gram + 线性 SVM）表现最强；深度学习中混合模型最优并与之
基本持平；深度模型召回率更高（漏判少），SVM 精确率更高（误报少）。

**数据洞察（重要）**：本数据集正常短信类几乎不含标点符号（含“，”比例为 0.0%），
而垃圾短信 82.9% 含“，”。模型因此学到“带标点的模板句式 → 垃圾”这一（在数据集内
成立、在真实场景有偏的）规律。演示时正常短信请使用口语化无标点文本；部署到真实
场景前建议补充带标点的正常短信样本微调。

## 演示

```bash
python src/predict.py --file demo_sms.txt \
    --model-path results/models/hybrid_char_best.pt
```

`demo_sms.txt` 中前 5 条为垃圾短信模板（模型输出 p≈1.0），后 5 条为口语化正常
短信（p≈0.0）。

## 参考资料与引用

- 数据集：[hrwhisper/SpamMessage](https://github.com/hrwhisper/SpamMessage)（中文垃圾短信识别）
- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
