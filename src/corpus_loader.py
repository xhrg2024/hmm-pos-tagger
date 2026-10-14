# -*- coding: utf-8 -*-
"""
语料库加载与解析模块
支持人民日报1998词性标注语料库、PKU-SEGPOS（OpenDataLab）语料库，
同时预留自定义语料库接口
"""

import json
import os
import re
from typing import List, Tuple, Optional


# 类型别名：一个词 = (词语, 词性标签)
WordTag = Tuple[str, str]
# 类型别名：一个句子 = [(词语1, 词性1), (词语2, 词性2), ...]
Sentence = List[WordTag]


def load_pku98_corpus(filepath: str) -> List[Sentence]:
    """
    加载人民日报1998词性标注语料库（2014兼容格式）

    原始格式示例（每行一个句子，词之间用制表符/空格分隔）：
        迈向/v\t充满/v\t希望/n\t的/u\t新/a\t世纪/n
        [中央/n\t人民/n\t广播/vn\t电台/n]/nt

    复合词 [word1/tag1 word2/tag2 ...]/outer_tag 会被拆分为多个词，
    每个子词继承外层标签 outer_tag。

    Parameters
    ----------
    filepath : str
        语料文件路径（如 199801-train.txt）

    Returns
    -------
    List[Sentence]
        解析后的句子列表，每个句子为 [(word, tag), ...] 的列表
    """
    sentences = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            sentence = _parse_line(line)
            if sentence:
                sentences.append(sentence)

    return sentences


# 人名相关标签（用于姓名合并）
NAME_TAGS = {'nr', 'nrf', 'nrg', 'nrd'}
# 中文人名最大长度（姓1字 + 名最多2字 = 3字，覆盖 95%+ 的中文人名）
MAX_NAME_LENGTH = 3


def _parse_line(line: str) -> Sentence:
    """
    解析单行语料为一个句子

    处理逻辑：
    1. 复合词 [word1/tag1 word2/tag2 ...]/outer_tag → 合并为一个大词，标签为 outer_tag
       例：[中央/n 人民/n 广播/vn 电台/n]/nt → (中央人民广播电台, nt)
    2. 普通词 word/tag → (word, tag)
    3. 相邻人名合并：连续的 nr/nrf/nrg 标签词合并为一个人名
       例：(王, nrf) (小明, nrg) → (王小明, nr)
    """
    raw_words = []

    # 用正则匹配复合词和普通词
    pattern = re.compile(r'\[([^\]]+)\]/(\w+)|(\S+?)/(\w+)')

    for match in pattern.finditer(line):
        if match.group(1) is not None:
            # 复合词：合并内部子词为一个词，使用外层标签
            inner_content = match.group(1)
            outer_tag = match.group(2)
            inner_words = inner_content.split()
            merged_word = ''
            for inner_word in inner_words:
                parts = inner_word.rsplit('/', 1)
                if len(parts) == 2:
                    word = parts[0]
                    if word and _is_valid_word(word):
                        merged_word += word
            if merged_word and _is_valid_tag(outer_tag):
                raw_words.append((merged_word, outer_tag))
        else:
            # 普通词
            word = match.group(3)
            tag = match.group(4)
            if word and tag and _is_valid_word(word) and _is_valid_tag(tag):
                raw_words.append((word, tag))

    # 合并相邻人名
    return _merge_adjacent_names(raw_words)


def _merge_adjacent_names(words: List[WordTag]) -> Sentence:
    """
    合并相邻的人名词

    如果连续多个词的标签都是人名相关标签（nr, nrf, nrg, nrd），
    且合并后长度不超过 MAX_NAME_LENGTH，则合并为一个词，标签统一为 nr。
    例：(王, nrf) (小明, nrg) → (王小明, nr)
    """
    if not words:
        return words

    result = []
    i = 0
    while i < len(words):
        word, tag = words[i]
        if tag in NAME_TAGS:
            # 开始收集连续的人名词
            merged = word
            j = i + 1
            while j < len(words) and words[j][1] in NAME_TAGS:
                candidate = merged + words[j][0]
                if len(candidate) <= MAX_NAME_LENGTH:
                    merged = candidate
                    j += 1
                else:
                    break
            result.append((merged, 'nr'))
            i = j
        else:
            result.append((word, tag))
            i += 1

    return result


def _is_valid_word(word: str) -> bool:
    """检查词语是否有效（排除标点和乱码）"""
    # 排除单独的 ] [ 等括号残留
    if word in (']', '[', '/', ''):
        return False
    # 至少包含一个中文字符或字母或数字
    return bool(re.search(r'[\u4e00-\u9fff\w]', word))


def _is_valid_tag(tag: str) -> bool:
    """检查词性标签是否有效（只保留合法标签）"""
    # 合法标签只包含字母（如 n, nr, ns, vn, ad 等）
    return bool(re.match(r'^[a-zA-Z]+$', tag))


def load_pku_segpos_jsonl(filepath: str) -> List[Sentence]:
    """
    加载 PKU-SEGPOS（OpenDataLab）词性标注语料库

    JSONL 格式，每行一个 JSON 对象：
    {
        "tokens": ["二", "○", "○", "○", "年", "贺", "词"],
        "span_list": [
            {"start": 0, "end": 4, "type": "t"},
            {"start": 5, "end": 6, "type": "n"}
        ]
    }

    tokens 是字符列表，span_list 中每个 span 用 start/end 索引指定一个词，
    type 为词性标签。

    Parameters
    ----------
    filepath : str
        JSONL 文件路径（如 train.jsonline）

    Returns
    -------
    List[Sentence]
        解析后的句子列表
    """
    sentences = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            tokens = data['tokens']
            span_list = data['span_list']

            # 按 start 排序，确保顺序正确
            span_list.sort(key=lambda s: s['start'])

            sentence = []
            prev_end = -1
            for span in span_list:
                start = span['start']
                end = span['end']  # inclusive
                tag = span['type']
                # tokens[start:end+1] 组成一个词
                word = ''.join(tokens[start:end + 1])
                # 处理嵌套标签（如 n]nt → 取专名标签 nt）
                if ']' in tag:
                    tag = tag.split(']')[-1]
                if not word or not _is_valid_tag(tag):
                    prev_end = end
                    continue

                # 合并相邻同标签的 span（人名标签除外，人名由后续 _merge_adjacent_names 处理）
                if (sentence and sentence[-1][1] == tag
                        and start == prev_end + 1
                        and tag not in NAME_TAGS):
                    # 非人名标签，正常合并
                    sentence[-1] = (sentence[-1][0] + word, tag)
                else:
                    sentence.append((word, tag))
                prev_end = end

            # 人名不做 span 合并，保持原始粒度
            # （PKU-SEGPOS 将姓名拆为姓/名两个 span，在连续多人名时无法可靠判断边界）
            # PKU98 的人名已经是整体标注，不需要合并

            if sentence:
                sentences.append(sentence)

    return sentences


def load_all_corpus(base_dir: str = None) -> List[Sentence]:
    """
    加载所有可用语料并合并

    依次加载：
    1. 人民日报 1998年1月（PKU98）
    2. PKU-SEGPOS（OpenDataLab，2000年数据）

    Parameters
    ----------
    base_dir : str
        项目根目录，默认为 src/ 的上级目录

    Returns
    -------
    List[Sentence]
        合并后的全部句子
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    all_sentences = []

    # 1. 加载 PKU98（1998年1月）
    pku98_train = os.path.join(base_dir, 'data', 'raw', 'pku98', 'pku98', '199801-train.txt')
    if os.path.exists(pku98_train):
        sentences = load_pku98_corpus(pku98_train)
        print(f"[PKU98] 加载 {len(sentences)} 句")
        all_sentences.extend(sentences)

    # 2. 加载 PKU-SEGPOS（2000年数据）
    segpos_dir = os.path.join(base_dir, 'data', 'raw', 'pku_segpos', 'PKU-SEGPOS', 'extracted')
    for filename in ['train.jsonline', 'dev.jsonline']:
        filepath = os.path.join(segpos_dir, filename)
        if os.path.exists(filepath):
            sentences = load_pku_segpos_jsonl(filepath)
            print(f"[PKU-SEGPOS/{filename}] 加载 {len(sentences)} 句")
            all_sentences.extend(sentences)

    print(f"\n合计: {len(all_sentences)} 句")
    return all_sentences


def load_custom_corpus(filepath: str, separator: str = '/',
                       sentence_delimiter: str = '\n') -> List[Sentence]:
    """
    加载自定义词性标注语料库（预留接口）

    支持的格式：
    - 每行一个句子
    - 词之间用空格/制表符分隔
    - 每个词格式为 word{separator}tag

    Parameters
    ----------
    filepath : str
        语料文件路径
    separator : str
        词与标签之间的分隔符，默认为 '/'
    sentence_delimiter : str
        句子之间的分隔符，默认为换行符

    Returns
    -------
    List[Sentence]
        解析后的句子列表
    """
    sentences = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for line in content.split(sentence_delimiter):
        line = line.strip()
        if not line:
            continue

        sentence = []
        tokens = line.split()
        for token in tokens:
            parts = token.rsplit(separator, 1)
            if len(parts) == 2:
                word, tag = parts
                if word and tag:
                    sentence.append((word, tag))

        if sentence:
            sentences.append(sentence)

    return sentences


def split_train_test(sentences: List[Sentence],
                     train_ratio: float = 0.9,
                     seed: int = 42) -> Tuple[List[Sentence], List[Sentence]]:
    """
    将语料划分为训练集和测试集

    Parameters
    ----------
    sentences : List[Sentence]
        全部句子
    train_ratio : float
        训练集比例，默认 0.9
    seed : int
        随机种子

    Returns
    -------
    Tuple[List[Sentence], List[Sentence]]
        (训练集句子列表, 测试集句子列表)
    """
    import random
    random.seed(seed)

    indices = list(range(len(sentences)))
    random.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_set = [sentences[i] for i in train_indices]
    test_set = [sentences[i] for i in test_indices]

    return train_set, test_set


def corpus_statistics(sentences: List[Sentence]) -> dict:
    """
    统计语料库基本信息

    Parameters
    ----------
    sentences : List[Sentence]
        解析后的句子列表

    Returns
    -------
    dict
        包含句子数、总词数、词表大小、标签集合等信息的字典
    """
    total_words = 0
    vocab = set()
    tags = set()
    tag_counts = {}

    for sentence in sentences:
        for word, tag in sentence:
            total_words += 1
            vocab.add(word)
            tags.add(tag)
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return {
        'num_sentences': len(sentences),
        'total_words': total_words,
        'vocab_size': len(vocab),
        'num_tags': len(tags),
        'tags': sorted(tags),
        'tag_counts': tag_counts,
        'avg_sentence_length': total_words / max(len(sentences), 1)
    }


if __name__ == '__main__':
    # 测试语料加载
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, 'data', 'raw', 'pku98', 'pku98', '199801-train.txt')

    if os.path.exists(train_file):
        print(f"正在加载语料: {train_file}")
        sentences = load_pku98_corpus(train_file)
        stats = corpus_statistics(sentences)

        print(f"\n===== 语料库统计信息 =====")
        print(f"句子数量: {stats['num_sentences']}")
        print(f"总词数:   {stats['total_words']}")
        print(f"词表大小: {stats['vocab_size']}")
        print(f"词性标签数: {stats['num_tags']}")
        print(f"平均句长: {stats['avg_sentence_length']:.1f}")
        print(f"\n词性标签列表:")
        for tag in stats['tags']:
            count = stats['tag_counts'][tag]
            print(f"  {tag:6s} : {count:6d} ({count/stats['total_words']*100:.2f}%)")

        print(f"\n前3个句子示例:")
        for i, sent in enumerate(sentences[:3]):
            print(f"  句子{i+1}: {sent[:8]}...")
    else:
        print(f"语料文件不存在: {train_file}")
