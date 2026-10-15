# 基于隐马尔可夫模型（HMM）的中文词性标注系统

## 项目简介

本项目实现了一个完整的中文词性标注系统，核心基于隐马尔可夫模型（HMM）。系统从语料库出发，完成数据预处理、模型训练、Viterbi 解码预测的完整流程，并提供终端交互应用和 Jupyter Notebook 两种使用方式。

**核心功能**：输入任意中文句子 → 自动分词 → 输出词性标注序列

**技术方案**：
- 分词：jieba 分词
- 词性标注：自研 HMM 模型（从语料中统计三要素 + Viterbi 解码）
- 训练语料：人民日报 1998 年 1 月 + 2000 年词性标注数据，共约 49,000 句、258 万词

---

## 文件结构说明

```
文本挖掘技术/
│
├── hmm_pos_tagging.ipynb      # Jupyter Notebook 完整流程（推荐入口）
├── app.py                      # 终端交互应用主程序
├── 启动词性标注.bat             # 双击启动终端应用
├── README.md                   # 本文件
├── 词性对照表.txt               # PKU 词性标注集对照表
│
├── src/                        # 源代码目录
│   ├── corpus_loader.py        # 语料加载与解析模块
│   ├── preprocess.py           # 数据预处理脚本（合并、清洗、保存）
│   ├── hmm_pos_tagger.py       # HMM 三要素统计模块
│   ├── viterbi_decoder.py      # Viterbi 解码器
│   ├── segmenter.py            # jieba 分词封装模块
│   └── oov_handler.py          # 未登录词（OOV）处理模块
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始语料（下载后未修改）
│   │   ├── pku98/              # 人民日报 1998 年 1 月语料
│   │   │   └── pku98/
│   │   │       ├── 199801-train.txt   # 训练集（17,524 句）
│   │   │       ├── 199801-test.txt    # 测试集（1,949 句）
│   │   │       ├── 199801.txt         # 完整语料
│   │   │       └── README.md          # 语料说明
│   │   └── pku_segpos/         # PKU-SEGPOS 语料（2000 年数据）
│   │       └── PKU-SEGPOS/extracted/
│   │           ├── train.jsonline     # 训练集（31,772 句）
│   │           ├── dev.jsonline       # 测试集（5,204 句）
│   │           └── test_public.jsonline # 公开测试集（无标注，未使用）
│   │
│   └── processed/              # 预处理后的数据
│       ├── train.txt           # 合并后的训练集（49,296 句，20.9 MB）
│       ├── test.txt            # 合并后的测试集（7,153 句，3.1 MB）
│       ├── hmm_model.json      # 训练好的 HMM 模型参数（301.9 MB）
│       ├── meta.json           # 数据元信息（来源、统计、标签列表）
│       └── tags.txt            # 词性标签频次表（辅助词典）
│
└── venv/                       # Python 虚拟环境（不提交）
```

---

## 源代码说明

### `src/corpus_loader.py` — 语料加载与解析

负责加载两个数据源的原始语料，并统一解析为 `[(词语, 标签), ...]` 格式。

| 函数 | 功能 |
|------|------|
| `load_pku98_corpus()` | 加载 PKU98 格式语料（`word/POS` 格式） |
| `load_pku_segpos_jsonl()` | 加载 PKU-SEGPOS 格式语料（JSONL 格式） |
| `load_all_corpus()` | 合并加载所有数据源 |
| `corpus_statistics()` | 统计语料库信息（句数、词数、标签分布等） |
| `split_train_test()` | 划分训练集和测试集 |

**关键处理逻辑**：
- **复合词合并**：`[中央/n 人民/n 广播/vn 电台/n]/nt` → `中央人民广播电台/nt`（合并而非拆分）
- **嵌套标签处理**：`n]nt` → 取专名标签 `nt`
- **脏数据过滤**：过滤乱码标签和无效词语

### `src/preprocess.py` — 数据预处理

将两个数据源合并、打乱、保存为统一格式。运行命令：
```bash
python src/preprocess.py
```

### `src/hmm_pos_tagger.py` — HMM 三要素统计

从训练语料中统计 HMM 三个核心概率：

| 要素 | 含义 | 计算方式 |
|------|------|----------|
| **π（初始概率）** | 句首为某标签的概率 | `P(tag) = (start_count + 1) / (N + \|tags\|)` |
| **A（转移概率）** | 标签 i 后接标签 j 的概率 | `P(j\|i) = (trans_count + 1) / (count_i + \|tags\|)` |
| **B（观测概率）** | 标签下出现某词的概率 | `P(word\|tag) = (emit_count + 1) / (count_tag + \|vocab\|)` |

采用**拉普拉斯（加一）平滑**处理未出现事件，所有概率以**对数形式**存储防止数值下溢。

### `src/viterbi_decoder.py` — Viterbi 解码

实现 Viterbi 动态规划算法，给定分词后的词语序列，求最可能的标签序列。

**算法流程**：
1. **初始化**：`δ[0][j] = log π(j) + log B(word₀ | tag_j)`
2. **递推**：`δ[t][j] = max_i(δ[t-1][i] + log A(j|i)) + log B(word_t | tag_j)`
3. **回溯**：通过 ψ 指针恢复最优路径

### `src/segmenter.py` — jieba 分词封装

封装 jieba 分词功能，提供统一接口：

| 函数 | 功能 |
|------|------|
| `segment(sentence)` | 分词，返回词语列表 |
| `segment_with_pos(sentence)` | 分词 + jieba 自带词性标注（供对比） |
| `segment_batch(sentences)` | 批量分词 |

### `src/oov_handler.py` — 未登录词（OOV）处理

实现三种 OOV 处理策略，提升未登录词的标注准确率：

| 类 | 策略 | 原理 |
|------|------|------|
| `CharFeatureEstimator` | 字符特征 | 统计字符-标签分布，通过字符组合估计 OOV 词的发射概率 |
| `RuleBasedTagger` | 规则匹配 | 根据词的表面形式（数字、标点、地名后缀等）直接判定标签 |
| `TagAvgSmoother` | 标签平均平滑 | 基于标签先验概率和不重复词数计算 OOV 概率分布 |
| `OOVHandler` | 综合加权 | 融合三种策略，规则权重 3.0 + 字符 2.0 + 平均 1.0 |

---

## 语料库说明

本项目使用两个数据源，均来自北京大学计算语言学研究所：

| 数据源 | 年份 | 格式 | 训练句数 | 来源 |
|--------|------|------|----------|------|
| PKU98 | 1998年1月 | `word/POS` 制表符分隔 | 17,524 | HanLP 镜像 |
| PKU-SEGPOS | 2000年 | JSONL（tokens + span_list） | 31,772 | OpenDataLab |

**合并后规模**：训练集 49,296 句 / 258 万词 / 57 种标签，测试集 7,153 句 / 38 万词。

**词性标签体系**：采用北大词性标注集（PKU POS Tagset），主要标签包括：

| 标签 | 含义 | 标签 | 含义 |
|------|------|------|------|
| n | 名词 | v | 动词 |
| a | 形容词 | d | 副词 |
| nr | 人名 | ns | 地名 |
| nt | 机构名 | nz | 其他专名 |
| p | 介词 | r | 代词 |
| m | 数词 | q | 量词 |
| u | 助词 | c | 连词 |
| w | 标点 | t | 时间词 |

完整标签频次表见 `data/processed/tags.txt`。

---

## 辅助文件

| 文件 | 说明 |
|------|------|
| `词性对照表.txt` | PKU 词性标注集对照表，包含中文名、英文名、说明与示例 |
| `data/processed/tags.txt` | 词性标签频次表，每行 `标签\t频次` |
| `data/processed/meta.json` | 数据元信息，包含数据来源、训练/测试集规模、标签列表、标签分布等 |
| `data/processed/hmm_model.json` | 训练好的 HMM 模型参数，包含 log π、log A、log B |

---

## 运行步骤

### 环境准备

```bash
# Python 版本：3.12+
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install numpy jupyter jieba requests matplotlib
```

### 方式一：Jupyter Notebook（推荐）

打开 `hmm_pos_tagging.ipynb`，从上到下依次运行所有 cell。Notebook 包含完整流程：

| 步骤 | 功能 | 对应模块 |
|------|------|----------|
| 第一步 | 语料加载与预处理 | `src/corpus_loader.py`、`src/preprocess.py` |
| 第二步 | HMM 三要素统计 | `src/hmm_pos_tagger.py` |
| 第三步 | Viterbi 解码 | `src/viterbi_decoder.py` |
| 第四步 | 模型评估 | Notebook 内实现 |
| 第五步 | 未登录词（OOV）处理优化 | `src/oov_handler.py` |
| 第六步 | 端到端标注（jieba 分词 + HMM） | `src/segmenter.py` |

### 方式二：终端交互应用

**双击** `启动词性标注.bat`，或在终端执行：

```bash
python app.py
```

输入任意中文句子，输出分词和词性标注结果。输入 `q` 退出。

### 方式三：重新训练模型

如需重新预处理数据并训练模型：

```bash
# 1. 数据预处理（合并两个数据源）
python src/preprocess.py

# 2. 统计 HMM 三要素并保存模型
python src/hmm_pos_tagger.py
```

---

## 模型评估

### 基线模型

在测试集（7,153 句，38 万词）上的评估结果：

| 指标 | 数值 |
|------|------|
| 整体准确率 | **88.78%** |
| 未登录词准确率 | 44.63% |

### OOV 优化效果

通过第五步的三种优化策略（字符特征、规则匹配、标签平均平滑），OOV 准确率显著提升：

| 策略 | OOV 准确率 |
|------|------------|
| 基线（无 OOV 处理） | 41.84% ± 12.18% |
| 综合策略（三种加权） | **76.46% ± 5.06%** |

---

## 总结

本项目实现了一个完整的基于 HMM 的中文词性标注系统，主要工作包括：

1. **语料处理**：下载并解析人民日报 1998 年和 2000 年两个词性标注语料库，处理复合词合并、嵌套标签提取等复杂格式，合并为约 49,000 句训练集和 7,000 句测试集。

2. **HMM 模型训练**：从训练语料中统计 HMM 三要素（初始概率 π、转移概率 A、发射概率 B），采用拉普拉斯平滑和对数概率存储，模型参数保存为 JSON 格式。

3. **Viterbi 解码**：实现动态规划算法，给定分词后的词语序列，求最可能的词性标签序列。

4. **端到端标注**：封装 jieba 分词，实现从原始句子到词性标注的完整流水线，并与 jieba 自带词性标注进行对比。

5. **OOV 优化**：针对未登录词准确率低的 问题，实现三种优化策略（字符特征估计、规则匹配、标签平均平滑），通过综合加权将 OOV 准确率从 41.84% 提升至 76.46%。

6. **可视化与评估**：使用 5 折交叉验证评估各策略的稳定性，并生成可视化对比图。

系统提供 Jupyter Notebook 和终端交互两种使用方式，核心逻辑封装在 `src/` 目录下的独立模块中，Notebook 作为流程编排入口。

---

## 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| Python | 3.12+ | 运行环境 |
| jieba | 0.42+ | 中文分词 |
| jupyter | - | Notebook 运行环境 |
| requests | - | 语料下载 |
| matplotlib | - | OOV 优化可视化 |
