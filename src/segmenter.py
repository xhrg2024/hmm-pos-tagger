# -*- coding: utf-8 -*-
"""
中文分词模块
封装 jieba 分词功能，提供统一的接口供 HMM 词性标注流水线使用
"""
import jieba


def segment(sentence):
    """
    对中文句子进行分词

    Parameters
    ----------
    sentence : str
        原始中文句子，如 "我爱北京天安门"

    Returns
    -------
    list of str : 分词后的词语列表，如 ["我", "爱", "北京", "天安门"]
    """
    return list(jieba.cut(sentence))


def segment_with_pos(sentence):
    """
    对中文句子进行分词并返回 jieba 自带的词性标注（仅供参考）

    Parameters
    ----------
    sentence : str
        原始中文句子

    Returns
    -------
    list of (str, str) : [(词语, jieba词性), ...]
    """
    import jieba.posseg as pseg
    return [(w.word, w.flag) for w in pseg.cut(sentence)]


def segment_batch(sentences):
    """
    批量分词

    Parameters
    ----------
    sentences : list of str
        多个句子

    Returns
    -------
    list of list of str : 每句的分词结果
    """
    return [segment(sent) for sent in sentences]


def tag_raw_sentence(sentence, decoder):
    """
    端到端词性标注: 原始句子 -> jieba分词 -> HMM标注

    Parameters
    ----------
    sentence : str
        原始中文句子
    decoder : ViterbiDecoder
        HMM Viterbi 解码器实例

    Returns
    -------
    list of (str, str) : [(词语, 预测标签), ...]
    """
    words = segment(sentence)
    words = [w for w in words if w.strip()]
    return decoder.tag_sentence(words)


def tag_raw_sentence_str(sentence, decoder):
    """
    端到端标注, 返回格式化字符串

    Parameters
    ----------
    sentence : str
        原始中文句子
    decoder : ViterbiDecoder
        HMM Viterbi 解码器实例

    Returns
    -------
    str : "词1/标签1 词2/标签2 ..."
    """
    tagged = tag_raw_sentence(sentence, decoder)
    return ' '.join(f'{w}/{t}' for w, t in tagged)
