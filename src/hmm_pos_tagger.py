# -*- coding: utf-8 -*-
"""
从标注语料中统计 HMM 三要素

HMM 三要素：
  π - 初始状态概率：P(句首为某标签)
  A - 状态转移概率：P(标签_j | 标签_i)
  B - 观测概率：    P(词语 | 标签)
"""
import json
import math
import os
from collections import defaultdict


def load_processed_corpus(filepath):
    """加载预处理后的语料文件（word/tag 格式）"""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs = []
            for token in line.split():
                parts = token.rsplit('/', 1)
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
            if pairs:
                sentences.append(pairs)
    return sentences


def compute_hmm_statistics(sentences):
    """
    从标注语料中统计 HMM 三要素的频数和概率

    Parameters
    ----------
    sentences : list
        训练数据，每个句子是 [(词语, 标签), ...] 的列表

    Returns
    -------
    dict : 包含三要素的频数和概率
    """
    n_sentences = len(sentences)
    total_words = 0

    # ==================== 频数统计 ====================
    print("=" * 60)
    print("第一步：从语料中统计 HMM 三要素")
    print("=" * 60)
    print(f"\n语料规模: {n_sentences} 句")

    # --- 1. 初始状态频数 ---
    # 统计每个标签作为句首标签的次数
    start_count = defaultdict(int)
    for sent in sentences:
        if sent:
            start_count[sent[0][1]] += 1

    # --- 2. 转移频数 ---
    # 统计 tag_i → tag_j 的转移次数
    transition_count = defaultdict(int)  # (tag_i, tag_j) -> count
    tag_count = defaultdict(int)         # 每个标签出现的总次数

    # --- 3. 发射频数 ---
    # 统计 (tag, word) 共现次数
    emission_count = defaultdict(int)    # (tag, word) -> count

    # 收集所有标签和词汇
    tags = set()
    vocab = set()

    for sent in sentences:
        for i, (word, tag) in enumerate(sent):
            total_words += 1
            tags.add(tag)
            vocab.add(word)
            tag_count[tag] += 1
            emission_count[(tag, word)] += 1

            if i > 0:
                prev_tag = sent[i - 1][1]
                transition_count[(prev_tag, tag)] += 1

    print(f"总词数: {total_words}")
    print(f"标签数: {len(tags)}")
    print(f"词表大小: {len(vocab)}")

    # ==================== 计算概率 ====================
    print(f"\n计算概率（拉普拉斯平滑）...")
    tags_sorted = sorted(tags)
    n_tags = len(tags)
    vocab_size = len(vocab)

    # --- π：初始状态概率 ---
    # P(tag) = (start_count[tag] + 1) / (n_sentences + n_tags)
    log_pi = {}
    for tag in tags_sorted:
        count = start_count.get(tag, 0) + 1
        prob = count / (n_sentences + n_tags)
        log_pi[tag] = math.log(prob)

    # --- A：状态转移概率 ---
    # P(tag_j | tag_i) = (trans_count[tag_i, tag_j] + 1) / (tag_count[tag_i] + n_tags)
    log_A = {}
    for tag_i in tags_sorted:
        denom = tag_count[tag_i] + n_tags
        row = {}
        for tag_j in tags_sorted:
            count = transition_count.get((tag_i, tag_j), 0) + 1
            row[tag_j] = math.log(count / denom)
        log_A[tag_i] = row

    # --- B：观测概率 ---
    # P(word | tag) = (emission_count[tag, word] + 1) / (tag_count[tag] + vocab_size)
    log_B = {}
    for tag in tags_sorted:
        denom = tag_count[tag] + vocab_size
        b_row = {}
        for word in vocab:
            count = emission_count.get((tag, word), 0) + 1
            b_row[word] = math.log(count / denom)
        log_B[tag] = b_row

    # ==================== 统计结果展示 ====================
    print("\n" + "-" * 60)
    print("统计结果")
    print("-" * 60)

    # 初始概率 Top-10
    pi_items = sorted(log_pi.items(), key=lambda x: -x[1])
    print(f"\n【π 初始状态概率】Top-10:")
    for tag, lp in pi_items[:10]:
        raw = start_count.get(tag, 0)
        print(f"  {tag:6s}: P={math.exp(lp):.6f}  (句首出现 {raw} 次)")

    # 转移概率示例
    print(f"\n【A 状态转移概率】示例:")
    for tag_i in ['v', 'n', 'adj', 'd', 'p']:
        if tag_i not in log_A:
            continue
        row = [(tag_j, log_A[tag_i][tag_j]) for tag_j in tags_sorted]
        row.sort(key=lambda x: -x[1])
        top5 = ', '.join(f'{t}({math.exp(p):.3f})' for t, p in row[:5])
        print(f"  {tag_i:4s} → {top5}")

    # 发射概率示例
    print(f"\n【B 观测概率】示例:")
    for tag in ['n', 'v', 'a', 'nr', 'nt']:
        if tag not in log_B:
            continue
        row = [(w, log_B[tag][w]) for w in vocab]
        row.sort(key=lambda x: -x[1])
        top5 = ', '.join(f'{w}({math.exp(p):.4f})' for w, p in row[:5])
        print(f"  {tag:4s}: {top5}")

    return {
        'tags': tags_sorted,
        'vocab': vocab,
        'n_sentences': n_sentences,
        'total_words': total_words,
        'log_pi': log_pi,
        'log_A': log_A,
        'log_B': log_B,
    }


def save_model(stats, filepath):
    """保存模型参数到 JSON 文件"""
    model = {
        'tags': stats['tags'],
        'vocab_size': len(stats['vocab']),
        'n_sentences': stats['n_sentences'],
        'total_words': stats['total_words'],
        'log_pi': stats['log_pi'],
        'log_A': stats['log_A'],
        'log_B': stats['log_B'],
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n模型已保存: {filepath} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train.txt')
    model_path = os.path.join(base_dir, 'data', 'processed', 'hmm_model.json')

    # 加载训练语料
    print("加载训练语料...")
    sentences = load_processed_corpus(train_path)

    # 统计 HMM 三要素
    stats = compute_hmm_statistics(sentences)

    # 保存模型
    save_model(stats, model_path)
