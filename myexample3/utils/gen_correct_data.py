#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/8 12:10 下午
# @File  : gen_correct_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 判断单词是否应该被替换
from collections import Counter
import os
import random
import json

def static():
    """
    统计下
    Returns:
    """
    examples = []
    filename = "/opt/salt-daily-check/notes/paper_train.md"
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
    print(f"错误单词统计: {cnt}")
    length_sentence = Counter([len(i[1]) for i in examples])
    print(f"句子长度统计: {sorted(length_sentence.items())}")
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

def repair(train_rate=0.8, test_rate=0.1, dev_rate=0.1):
    """
    生成数据集dev.json test.json train.json, 生成labels
    Returns:
    """
    examples = static()
    dir_path = "../dataset/repair"
    label_file = os.path.join(dir_path, "labels.json")
    dev_file = os.path.join(dir_path, "dev.json")
    test_file = os.path.join(dir_path,"test.json")
    train_file = os.path.join(dir_path,"train.json")
    # 随机拆分句子为正样本，txt --> sentence1, sentence2
    # 随机拆分句子为负样本, txt1, txt2  ---> sentence1(from txt1), sentence2(from txt2)
    random.shuffle(examples)
    #拆分样本
    total = len(examples)
    if train_rate == 1:
        train_data = examples
        start_num = int(total*0.8)
        test_num = int(total*0.1)
        test_data = examples[start_num:start_num+test_num]
        dev_data = examples[start_num+test_num:]
    else:
        train_num = int(total*train_rate)
        test_num = int(total*test_rate)
        train_data = examples[:train_num]
        test_data = examples[train_num:train_num+test_num]
        dev_data = examples[train_num+test_num:]
    labels = sorted(list(set([i[3].strip() for i in examples])), key=len)
    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    with open(dev_file, 'w') as f:
        json.dump(dev_data, f)
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    with open(label_file, 'w') as f:
        json.dump(labels, f)
    print(f"训练集{len(train_data)}, 测试集{test_num}, 开发集{len(dev_data)}")
    print(f"标签数量{len(labels)}, 标签有:{labels}")

if __name__ == '__main__':
    # examples = static()
    # save_examples(examples)
    repair(train_rate=0.8)