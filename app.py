# -*- coding: utf-8 -*-
"""
HMM 中文词性标注 - 终端交互应用
输入任意中文句子，输出分词 + 词性标注结果
"""
import os
import sys

# 设置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from segmenter import segment
from viterbi_decoder import ViterbiDecoder

# 标签中文说明
TAG_DESC = {
    'n': '名词', 'nr': '人名', 'ns': '地名', 'nt': '机构名', 'nz': '其他专名',
    'v': '动词', 'vn': '名动词', 'vd': '副动词',
    'a': '形容词', 'an': '名形词', 'ad': '副形词',
    'd': '副词', 'p': '介词', 'r': '代词',
    'm': '数词', 'q': '量词', 'b': '区别词',
    'f': '方位词', 's': '处所词', 't': '时间词',
    'u': '助词', 'c': '连词', 'w': '标点',
    'j': '简称', 'l': '习用语', 'i': '成语',
    'k': '后接成分', 'h': '前接成分',
    'y': '语气词', 'e': '叹词', 'o': '拟声词',
    'Ng': '名语素', 'Vg': '动语素', 'Ag': '形语素',
}


def main():
    # 加载模型
    model_path = os.path.join(BASE_DIR, 'data', 'processed', 'hmm_model.json')
    if not os.path.exists(model_path):
        print('错误: 模型文件不存在，请先运行 hmm_pos_tagging.ipynb 训练模型')
        print(f'期望路径: {model_path}')
        input('\n按回车键退出...')
        return

    print('正在加载 HMM 模型...')
    decoder = ViterbiDecoder(model_path)
    print(f'模型加载完成（{len(decoder.tags)} 种标签）')
    print()
    print('=' * 60)
    print('  HMM 中文词性标注系统')
    print('  输入任意中文句子，输出分词和词性标注结果')
    print('  输入 q 或 quit 退出')
    print('=' * 60)

    while True:
        print()
        sentence = input('请输入句子: ').strip()

        if not sentence:
            continue
        if sentence.lower() in ('q', 'quit', 'exit'):
            print('再见！')
            break

        # 分词
        words = [w for w in segment(sentence) if w.strip()]

        # HMM 词性标注
        tagged = decoder.tag_sentence(words)

        # 输出结果
        print()
        print(f'  分词: {" / ".join(words)}')
        print()
        print(f'  标注: {" ".join(f"{w}/{t}" for w, t in tagged)}')
        print()

        # 逐词详解
        print('  逐词说明:')
        for word, tag in tagged:
            desc = TAG_DESC.get(tag, tag)
            print(f'    {word:<10s} → {tag:<4s} ({desc})')


if __name__ == '__main__':
    main()
