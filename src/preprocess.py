# -*- coding: utf-8 -*-
"""
数据预处理脚本
将 PKU98 和 PKU-SEGPOS 两个数据源合并、清洗，保存到 data/processed/
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from corpus_loader import (
    load_pku98_corpus,
    load_pku_segpos_jsonl,
    corpus_statistics
)

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)


def save_sentences(sentences, filepath):
    """将句子列表保存为 word/tag 格式（每行一句，词间空格）"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sent in sentences:
            line = ' '.join(f'{word}/{tag}' for word, tag in sent)
            f.write(line + '\n')


def load_sentences(filepath):
    """从 word/tag 格式文件加载句子列表"""
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


def main():
    print("=" * 60)
    print("数据预处理：合并 PKU98 + PKU-SEGPOS")
    print("=" * 60)

    # ===== 1. 加载各数据源 =====
    print("\n[1/4] 加载数据源...")

    # PKU98（1998年1月）
    pku98_train_path = os.path.join(RAW_DIR, 'pku98', 'pku98', '199801-train.txt')
    pku98_test_path = os.path.join(RAW_DIR, 'pku98', 'pku98', '199801-test.txt')
    pku98_train = load_pku98_corpus(pku98_train_path)
    pku98_test = load_pku98_corpus(pku98_test_path)
    print(f"  PKU98 训练集: {len(pku98_train)} 句")
    print(f"  PKU98 测试集: {len(pku98_test)} 句")

    # PKU-SEGPOS（2000年）
    segpos_dir = os.path.join(RAW_DIR, 'pku_segpos', 'PKU-SEGPOS', 'extracted')
    segpos_train = load_pku_segpos_jsonl(os.path.join(segpos_dir, 'train.jsonline'))
    segpos_dev = load_pku_segpos_jsonl(os.path.join(segpos_dir, 'dev.jsonline'))
    print(f"  PKU-SEGPOS 训练集: {len(segpos_train)} 句")
    print(f"  PKU-SEGPOS 开发集: {len(segpos_dev)} 句")

    # ===== 2. 合并数据 =====
    print("\n[2/4] 合并数据...")
    all_train = pku98_train + segpos_train
    all_test = pku98_test + segpos_dev
    print(f"  合并训练集: {len(all_train)} 句")
    print(f"  合并测试集: {len(all_test)} 句")

    # 打乱训练集顺序
    random.seed(42)
    random.shuffle(all_train)

    # ===== 3. 统计信息 =====
    print("\n[3/4] 统计信息...")
    train_stats = corpus_statistics(all_train)
    test_stats = corpus_statistics(all_test)

    print(f"\n  {'指标':<15s} {'训练集':>10s} {'测试集':>10s}")
    print(f"  {'-'*35}")
    print(f"  {'句子数':<15s} {train_stats['num_sentences']:>10d} {test_stats['num_sentences']:>10d}")
    print(f"  {'总词数':<15s} {train_stats['total_words']:>10d} {test_stats['total_words']:>10d}")
    print(f"  {'词表大小':<15s} {train_stats['vocab_size']:>10d} {test_stats['vocab_size']:>10d}")
    print(f"  {'标签数':<15s} {train_stats['num_tags']:>10d} {test_stats['num_tags']:>10d}")

    # ===== 4. 保存 =====
    print("\n[4/4] 保存到 data/processed/...")

    # 保存训练集和测试集
    train_path = os.path.join(PROC_DIR, 'train.txt')
    test_path = os.path.join(PROC_DIR, 'test.txt')
    save_sentences(all_train, train_path)
    save_sentences(all_test, test_path)
    print(f"  训练集: {train_path}")
    print(f"  测试集: {test_path}")

    # 保存元数据
    meta = {
        'sources': ['PKU98 (1998-01)', 'PKU-SEGPOS (2000)'],
        'train_sentences': len(all_train),
        'test_sentences': len(all_test),
        'train_words': train_stats['total_words'],
        'test_words': test_stats['total_words'],
        'vocab_size': train_stats['vocab_size'],
        'num_tags': train_stats['num_tags'],
        'tags': sorted(train_stats['tag_counts'].keys()),
        'tag_counts': train_stats['tag_counts'],
        'format': '每行一句，词间空格分隔，每个词为 word/tag 格式'
    }
    meta_path = os.path.join(PROC_DIR, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元数据: {meta_path}")

    # 保存标签列表
    tags_path = os.path.join(PROC_DIR, 'tags.txt')
    with open(tags_path, 'w', encoding='utf-8') as f:
        for tag, count in sorted(train_stats['tag_counts'].items(), key=lambda x: -x[1]):
            f.write(f"{tag}\t{count}\n")
    print(f"  标签表: {tags_path}")

    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)

    # 验证：重新加载并检查
    print("\n验证：重新加载已保存数据...")
    loaded_train = load_sentences(train_path)
    loaded_test = load_sentences(test_path)
    assert len(loaded_train) == len(all_train), f"训练集行数不匹配: {len(loaded_train)} vs {len(all_train)}"
    assert len(loaded_test) == len(all_test), f"测试集行数不匹配: {len(loaded_test)} vs {len(all_test)}"
    assert loaded_train[0] == all_train[0], "训练集内容不匹配"
    print(f"  训练集: {len(loaded_train)} 句 ✓")
    print(f"  测试集: {len(loaded_test)} 句 ✓")
    print(f"  前3个词: {loaded_train[0][:3]} ✓")


if __name__ == '__main__':
    main()
