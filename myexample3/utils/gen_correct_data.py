#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/8 12:10 下午
# @File  : gen_correct_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 判断单词是否应该被替换
from collections import Counter

def static():
    """
    统计下
    Returns:
    """
    examples = []
    filename = "paper_train.md"
    with open(filename, 'r') as f:
        lines = f.readlines()
        # 每5行一个样本,
        for i in range(0, len(lines), 5):
            eng_word = lines[i]
            sentence = lines[i+1]
            wrong_word = lines[i+2]
            correct_word = lines[i+3]
            blank = lines[i+4]
            if eng_word.strip() and sentence.strip() and wrong_word.strip() and correct_word.strip() and blank == '\n':
                examples.append([eng_word, sentence, wrong_word, correct_word, blank])
            else:
                print(f"第{i}行样本有问题，请检查")
                return
    print(f"共有样本数{len(examples)}")
    cnt = Counter([i[2] for i in examples])
    print(cnt)
    return examples

def save_examples(examples):
    # 每个样wrong_word的示例保留最多保留20个就可以了
    cnt = Counter([i[2] for i in examples])
    output = "train.txt"
    filter_examples = []
    for k,v in cnt.items():
        if v > 20:
            tmp = [i for i in examples if i[2] == k]
            filter_examples.extend(tmp[:20])
        else:
            filter_examples.extend([i for i in examples if i[2] == k])
    print(f"过滤后的样本数{len(filter_examples)}")
    with open(output, 'w') as f:
        for e in filter_examples:
            f.writelines(e)
    print(f"保存文件成功")
if __name__ == '__main__':
    examples = static()
    save_examples(examples)