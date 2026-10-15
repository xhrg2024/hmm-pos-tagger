# -*- coding: utf-8 -*-
"""
未登录词（OOV）处理模块
提供三种策略提升未登录词的词性标注准确率：
  5.1 基于字符特征的 OOV 估计
  5.2 基于规则的 OOV 处理
  5.3 基于标签条件平均发射概率的平滑策略
  5.4 交叉验证评估与可视化
"""
import re
import math
from collections import defaultdict


# ============================================================
# 5.1 基于字符特征的 OOV 估计
# ============================================================

class CharFeatureEstimator:
    """
    利用训练集中每个字符在各标签下的分布来估计 OOV 词的发射概率。

    原理：即使一个完整的词未见过，它的组成字符往往在某些标签下更常见。
    例如"省""市""县"在 ns（地名）下频次很高，
    因此包含这些字符的新词更可能是 ns。
    """

    def __init__(self, sentences):
        """
        从训练语料中统计字符-标签分布

        Parameters
        ----------
        sentences : list of list of (str, str)
            训练语料，每句为 [(词语, 标签), ...]
        """
        # char_tag_count[char][tag] = 该字符以该标签出现的次数
        self.char_tag_count = defaultdict(lambda: defaultdict(int))
        # tag_char_count[tag] = 该标签下所有字符的总出现次数
        self.tag_char_count = defaultdict(int)

        for sent in sentences:
            for word, tag in sent:
                for ch in word:
                    self.char_tag_count[ch][tag] += 1
                    self.tag_char_count[tag] += 1

        self.all_tags = set(self.tag_char_count.keys())

    def estimate_log_emit(self, word, tag):
        """
        基于字符特征估计 log P(word | tag)

        方法：将词中每个字符在该标签下的概率相乘（log 相加）
        P(word|tag) ≈ ∏ P(char_i | tag)
        P(char|tag) = char_tag_count[char][tag] / tag_char_count[tag]

        Parameters
        ----------
        word : str
            词语
        tag : str
            标签

        Returns
        -------
        float : log 发射概率
        """
        if tag not in self.tag_char_count:
            return math.log(1e-8)

        log_prob = 0.0
        for ch in word:
            count = self.char_tag_count[ch].get(tag, 0)
            total = self.tag_char_count[tag]
            # 拉普拉斯平滑
            prob = (count + 1) / (total + len(self.all_tags))
            log_prob += math.log(prob)

        return log_prob

    def get_best_tag(self, word):
        """
        仅基于字符特征，给出最可能的标签

        Parameters
        ----------
        word : str

        Returns
        -------
        str : 最可能的标签
        """
        best_tag = None
        best_score = -float('inf')
        for tag in self.all_tags:
            score = self.estimate_log_emit(word, tag)
            if score > best_score:
                best_score = score
                best_tag = tag
        return best_tag


# ============================================================
# 5.2 基于规则的 OOV 处理
# ============================================================

class RuleBasedTagger:
    """
    通过词的表面形式（字形模式）直接判定标签。
    规则优先级从高到低匹配，命中即返回。
    """

    # 规则列表：(正则模式, 标签, 说明)
    RULES = [
        # 纯数字
        (r'^\d+(\.\d+)?$', 'm', '纯数字→数词'),
        # 数字+量词模式（如 "第1"）
        (r'^第\d+', 'm', '第+数字→数词'),
        # 年份
        (r'^\d{4}年$', 't', 'xxxx年→时间词'),
        # 日期
        (r'^\d{1,2}月\d{1,2}[日号]$', 't', '日期→时间词'),

        # 标点符号
        (r'^[，。！？；：、\u201c\u201d\u2018\u2019（）《》【】…—.,!?;:\'"()\[\]{}-]+$', 'w', '标点'),

        # 地名后缀
        (r'.+(省|市|县|区|镇|乡|村|山|河|湖|海|岛)$', 'ns', '地名后缀→地名'),
        # 机构名后缀
        (r'.+(公司|集团|银行|学院|大学|学校|医院|委员会|办公室|研究所|总局|局|部|处|科)$', 'nt', '机构后缀→机构名'),
        # 人名：常见姓氏 + 1-2字
        (r'^[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于傅皮卞齐康伍余元卜顾孟平黄和穆萧尹][一二三四五六七八九]?[\u4e00-\u9fff]{0,2}$', 'nr', '姓氏+名→人名'),
    ]

    def __init__(self):
        # 预编译正则
        self._compiled = [(re.compile(pat), tag, desc) for pat, tag, desc in self.RULES]

    def predict(self, word):
        """
        尝试用规则匹配词语

        Parameters
        ----------
        word : str

        Returns
        -------
        str or None : 匹配到的标签，未匹配返回 None
        """
        for pattern, tag, desc in self._compiled:
            if pattern.match(word):
                return tag
        return None

    def get_log_emit(self, word, tag):
        """
        规则匹配成功时返回较高的发射概率，
        未匹配返回 None 表示不使用规则
        """
        predicted = self.predict(word)
        if predicted == tag:
            return math.log(0.5)  # 规则匹配，给较高概率
        elif predicted is not None:
            return math.log(1e-6)  # 规则匹配到其他标签，给低概率
        return None  # 无规则命中，不使用规则


# ============================================================
# 5.3 基于标签条件平均发射概率的平滑策略
# ============================================================

class TagAvgSmoother:
    """
    用标签的条件平均发射概率替代固定的 1e-8。

    原理：OOV 词属于某标签的概率取决于两个因素：
    1. 词性占比 P(tag)：占比越高，OOV 越可能属于它
    2. 不重复词数 unique(tag)：同等占比下，词汇量越大的标签越能"接纳"新词

    公式：OOV_score(tag) = P(tag) × unique_words(tag)
    归一化后得到各标签的 OOV 概率分布。
    """

    def __init__(self, sentences, model):
        """
        Parameters
        ----------
        sentences : list
            训练语料，用于统计 P(tag) 和 unique_words(tag)
        model : dict
            hmm_model.json 中的模型参数
        """
        self.tag_avg_log_prob = {}

        # 从训练语料统计每个标签的 token 数和 unique word 数
        tag_token_count = defaultdict(int)  # 标签总出现次数
        tag_unique_words = defaultdict(set)  # 标签下的不重复词集合
        total_tokens = 0

        for sent in sentences:
            for word, tag in sent:
                tag_token_count[tag] += 1
                tag_unique_words[tag].add(word)
                total_tokens += 1

        # 计算 OOV_score(tag) = P(tag) × unique_words(tag)
        raw_scores = {}
        for tag in model['tags']:
            p_tag = tag_token_count.get(tag, 0) / max(total_tokens, 1)
            n_unique = len(tag_unique_words.get(tag, set()))
            raw_scores[tag] = p_tag * n_unique

        # 归一化为概率分布
        total_score = sum(raw_scores.values())
        for tag in model['tags']:
            if total_score > 0:
                prob = raw_scores[tag] / total_score
            else:
                prob = 1.0 / len(model['tags'])
            self.tag_avg_log_prob[tag] = math.log(max(prob, 1e-20))

    def get_log_emit(self, word, tag):
        """
        获取标签条件平均 log 发射概率

        Parameters
        ----------
        word : str
        tag : str

        Returns
        -------
        float : log 发射概率
        """
        return self.tag_avg_log_prob.get(tag, math.log(1e-8))


# ============================================================
# 综合 OOV 处理器
# ============================================================

class OOVHandler:
    """
    综合三种策略的 OOV 处理器。
    优先级：规则(5.2) > 字符特征(5.1) > 标签平均(5.3)
    """

    def __init__(self, sentences, model, use_rules=True, use_char=True, use_avg=True):
        """
        Parameters
        ----------
        sentences : list
            训练语料
        model : dict
            模型参数
        use_rules : bool
            是否启用规则(5.2)
        use_char : bool
            是否启用字符特征(5.1)
        use_avg : bool
            是否启用标签平均(5.3)
        """
        self.use_rules = use_rules
        self.use_char = use_char
        self.use_avg = use_avg

        if use_rules:
            self.rule_tagger = RuleBasedTagger()
        if use_char:
            self.char_estimator = CharFeatureEstimator(sentences)
        if use_avg:
            self.avg_smoother = TagAvgSmoother(sentences, model)

        self._fallback = math.log(1e-8)

    def get_log_emit(self, tag, word, is_oov=True):
        """
        获取 OOV 词的 log 发射概率，综合三种策略

        Parameters
        ----------
        tag : str
        word : str
        is_oov : bool
            是否为未登录词

        Returns
        -------
        float : log 发射概率
        """
        if not is_oov:
            return self._fallback

        scores = []

        # 策略 5.2：规则匹配（权重最高）
        if self.use_rules:
            rule_score = self.rule_tagger.get_log_emit(word, tag)
            if rule_score is not None:
                scores.append(('rule', rule_score, 3.0))  # 权重 3.0

        # 策略 5.1：字符特征
        if self.use_char:
            char_score = self.char_estimator.estimate_log_emit(word, tag)
            scores.append(('char', char_score, 2.0))  # 权重 2.0

        # 策略 5.3：标签平均
        if self.use_avg:
            avg_score = self.avg_smoother.get_log_emit(word, tag)
            scores.append(('avg', avg_score, 1.0))  # 权重 1.0

        if not scores:
            return self._fallback

        # 加权平均（log 空间不能直接加权，转为概率空间）
        total_weight = sum(w for _, _, w in scores)
        log_prob = 0.0
        for _, s, w in scores:
            log_prob += (w / total_weight) * s

        return log_prob


# ============================================================
# 快速评估工具
# ============================================================

def quick_evaluate(decoder, test_data, n_samples=100):
    """
    快速评估解码器性能（少量样本）

    Parameters
    ----------
    decoder : ViterbiDecoder
        解码器实例
    test_data : list
        测试集，每句为 [(词语, 标签), ...]
    n_samples : int
        评估样本数，默认 100

    Returns
    -------
    tuple : (整体准确率, OOV准确率)
    """
    total = correct = oov_total = oov_correct = 0
    
    for sent in test_data[:n_samples]:
        words = [w for w, t in sent]
        gold_tags = [t for w, t in sent]
        pred_tags = decoder.decode(words)
        for word, gold, pred in zip(words, gold_tags, pred_tags):
            total += 1
            if gold == pred:
                correct += 1
            if word not in decoder.log_B.get(gold, {}):
                oov_total += 1
                if gold == pred:
                    oov_correct += 1
    
    acc = correct / total * 100 if total > 0 else 0
    oov_acc = oov_correct / oov_total * 100 if oov_total > 0 else 0
    
    return acc, oov_acc


# ============================================================
# 5.4 交叉验证评估与可视化
# ============================================================

def evaluate_with_kfold(decoders, test_data, n_folds=5):
    """
    使用 k 折交叉验证评估多个解码器的稳定性

    Parameters
    ----------
    decoders : dict
        {名称: ViterbiDecoder 实例}
    test_data : list
        测试集，每句为 [(词语, 标签), ...]
    n_folds : int
        折数，默认 5

    Returns
    -------
    dict : {名称: {'acc': [各折准确率], 'oov_acc': [各折OOV准确率]}}
    """
    import numpy as np

    fold_size = len(test_data) // n_folds
    results = {name: {'acc': [], 'oov_acc': []} for name in decoders}

    print(f'5 折交叉验证（全测试集 {len(test_data)} 句，每折约 {fold_size} 句）:')
    print('=' * 80)

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(test_data)
        fold_data = test_data[start:end]

        print(f'  Fold {fold+1}/{n_folds}（第 {start+1}~{end} 句）...')

        for name, dec in decoders.items():
            total = correct = oov_total = oov_correct = 0
            for sent in fold_data:
                words = [w for w, t in sent]
                gold_tags = [t for w, t in sent]
                pred_tags = dec.decode(words)
                for word, gold, pred in zip(words, gold_tags, pred_tags):
                    total += 1
                    if gold == pred:
                        correct += 1
                    if word not in dec.log_B.get(gold, {}):
                        oov_total += 1
                        if gold == pred:
                            oov_correct += 1
            acc = correct / total * 100 if total > 0 else 0
            oov_acc = oov_correct / oov_total * 100 if oov_total > 0 else 0
            results[name]['acc'].append(acc)
            results[name]['oov_acc'].append(oov_acc)

    # 打印结果表
    print()
    print(f'{"策略":<20s} {"整体准确率(mean±std)":>25s} {"OOV准确率(mean±std)":>25s}')
    print('-' * 80)
    for name in decoders:
        acc_mean = np.mean(results[name]['acc'])
        acc_std = np.std(results[name]['acc'])
        oov_mean = np.mean(results[name]['oov_acc'])
        oov_std = np.std(results[name]['oov_acc'])
        print(f'{name:<20s} {acc_mean:>8.2f}% ± {acc_std:<5.2f}%     {oov_mean:>8.2f}% ± {oov_std:<5.2f}%')

    return results


def plot_oov_comparison(results, n_folds=5, save_path='oov_comparison.png'):
    """
    绘制 OOV 策略对比可视化图（双子图：整体准确率 + OOV 准确率）

    Parameters
    ----------
    results : dict
        evaluate_with_kfold() 的返回值
    n_folds : int
        折数
    save_path : str
        图片保存路径
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # 配置中文字体（Windows 系统）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong', 'STFangsong']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n_folds)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
    markers = ['o', 's', '^', 'D', 'v']

    # 左图：整体准确率
    ax = axes[0]
    for idx, (name, color, marker) in enumerate(zip(results.keys(), colors, markers)):
        ax.plot(x, results[name]['acc'], color=color, marker=marker, linewidth=2,
                markersize=8, label=name)
        mean_val = np.mean(results[name]['acc'])
        ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.4)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('整体准确率 (%)', fontsize=12)
    ax.set_title('整体准确率 — 5 折交叉验证', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右图：OOV 准确率
    ax = axes[1]
    for idx, (name, color, marker) in enumerate(zip(results.keys(), colors, markers)):
        ax.plot(x, results[name]['oov_acc'], color=color, marker=marker, linewidth=2,
                markersize=8, label=name)
        mean_val = np.mean(results[name]['oov_acc'])
        ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.4)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('OOV 准确率 (%)', fontsize=12)
    ax.set_title('未登录词(OOV)准确率 — 5 折交叉验证', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print()
    print(f'可视化图片已保存为 {save_path}')
