# -*- coding: utf-8 -*-
"""
Viterbi 解码器
实现 HMM 解码：输入分词后的句子，输出最可能的词性序列
"""
import json
import math


class ViterbiDecoder:
    """Viterbi 算法解码器"""

    def __init__(self, model_path):
        """
        加载 HMM 模型参数

        Parameters
        ----------
        model_path : str
            hmm_model.json 文件路径
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            model = json.load(f)

        self.tags = model['tags']
        self.log_pi = model['log_pi']
        self.log_A = model['log_A']
        self.log_B = model['log_B']
        self.vocab_size = model['vocab_size']

        # 未登录词的默认 log 发射概率
        # 使用拉普拉斯平滑中的未登录词概率：1 / (tag_count + vocab_size)
        # 简化为一个较小的固定值
        self._unk_log_prob = math.log(1e-8)

    def decode(self, words):
        """
        Viterbi 解码：给定词语序列，求最可能的标签序列

        算法流程：
        1. 初始化：δ[0][j] = log π(j) + log B(word_0 | tag_j)
        2. 递推：δ[t][j] = max_i(δ[t-1][i] + log A(tag_j | tag_i)) + log B(word_t | tag_j)
        3. 终止：取 max_j δ[T-1][j]
        4. 回溯：通过 ψ 指针回溯最优路径

        Parameters
        ----------
        words : list of str
            分词后的词语列表

        Returns
        -------
        list of str : 最可能的标签序列
        """
        if not words:
            return []

        T = len(words)
        tags = self.tags
        n_tags = len(tags)

        # δ[t][j] = 到时刻 t、状态为 tags[j] 的最大 log 概率
        delta = [[0.0] * n_tags for _ in range(T)]
        # ψ[t][j] = 回溯指针：时刻 t 状态 j 的最优前驱状态索引
        psi = [[0] * n_tags for _ in range(T)]

        # ---- 初始化（t=0）----
        word = words[0]
        for j, tag in enumerate(tags):
            log_emit = self._get_log_emit(tag, word)
            delta[0][j] = self.log_pi.get(tag, self._unk_log_prob) + log_emit

        # ---- 递推（t=1..T-1）----
        for t in range(1, T):
            word = words[t]
            for j, tag_j in enumerate(tags):
                best_val = -float('inf')
                best_prev = 0
                log_emit = self._get_log_emit(tag_j, word)

                for i, tag_i in enumerate(tags):
                    val = delta[t - 1][i] + self.log_A[tag_i].get(tag_j, self._unk_log_prob)
                    if val > best_val:
                        best_val = val
                        best_prev = i

                delta[t][j] = best_val + log_emit
                psi[t][j] = best_prev

        # ---- 终止：找最优终态 ----
        best_last = 0
        best_val = delta[T - 1][0]
        for j in range(1, n_tags):
            if delta[T - 1][j] > best_val:
                best_val = delta[T - 1][j]
                best_last = j

        # ---- 回溯：恢复最优路径 ----
        result = [None] * T
        result[T - 1] = tags[best_last]
        for t in range(T - 2, -1, -1):
            best_last = psi[t + 1][best_last]
            result[t] = tags[best_last]

        return result

    def _get_log_emit(self, tag, word):
        """获取 log P(word | tag)，处理未登录词"""
        if tag in self.log_B and word in self.log_B[tag]:
            return self.log_B[tag][word]
        # 未登录词：返回一个较小的默认概率
        return self._unk_log_prob

    def tag_sentence(self, words):
        """
        对分词后的句子进行词性标注

        Parameters
        ----------
        words : list of str
            分词后的词语列表

        Returns
        -------
        list of (str, str) : [(词语, 标签), ...]
        """
        tags = self.decode(words)
        return list(zip(words, tags))

    def tag_sentence_str(self, words, separator=' '):
        """
        对分词后的句子进行词性标注，返回格式化字符串

        Parameters
        ----------
        words : list of str
            分词后的词语列表
        separator : str
            词之间的分隔符

        Returns
        -------
        str : "词1/标签1 词2/标签2 ..."
        """
        tagged = self.tag_sentence(words)
        return separator.join(f'{word}/{tag}' for word, tag in tagged)
