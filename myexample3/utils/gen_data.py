#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/1/27 10:30 上午
# @File  : gen_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 生成二分类的数据集, 上下句是连贯的，那么值为1，否则为0
from pathlib import Path
import re
import random, os
import json
from tqdm import tqdm
from collections import Counter

def search_and_read(path='/opt/nlp/pycorrect'):
    data = []
    for fpath in Path(path).rglob('*.mix'):
        with open(fpath) as f:
            for line in f:
                #匹配英文，并且，单词数大于10
                if len(line.split()) > 20 and not re.search("[\u4e00-\u9fa5]", line):
                    # 按逗号和句号分隔空格分隔
                    line_split = re.split('[,.]', line)
                    for sentence in line_split:
                        if len(sentence) >10:
                            #按空格分隔
                            tokens = sentence.split()
                            if len(tokens) > 4:
                                #至少这个句子包含5个词
                                data.append(tokens)
    cnt = Counter([len(d) for d in data])
    print(f"生成文件行数{len(data)}, 长度最多的个数是{cnt.most_common(1)[0][1]},长度是{cnt.most_common(1)[0][0]}, 样本的长度规则是{cnt}")
    return data

def split_train_test(data):
    """
    生成正负样本，保存到json文件, 生成8:1:1的样本，保存到train.json, test.json, dev.json
    Args:
        data: [txt1,txt2,txt3,...]
    Returns:
    """
    # 正样本样本组成[[sen1,sen2,yes]... ]
    positive = []
    negative = []
    dir_path = "../dataset"
    dev_file = os.path.join(dir_path, "dev.json")
    test_file = os.path.join(dir_path,"test.json")
    train_file = os.path.join(dir_path,"train.json")
    # 随机拆分句子为正样本，txt --> sentence1, sentence2
    # 随机拆分句子为负样本, txt1, txt2  ---> sentence1(from txt1), sentence2(from txt2)
    def gen_neg_smaple(data):
        """
        Args:
            data: 所有样本
        Returns:返回一个句子的的随机部分，作为负样本
        """
        # 负样本,随机组合负样本
        neg_one = random.choice(data)
        neg_len = len(neg_one)
        neg_split = random.randrange(1, neg_len - 1)
        neg_half_sentence = " ".join(neg_one[neg_split:])
        return neg_half_sentence
    for one in tqdm(data, desc="样本生成中: "):
        #正样本
        pos_len = len(one)
        #随机取一个分隔点
        pos_split = random.randrange(1,pos_len-1)
        neg_half_sentence1 = gen_neg_smaple(data)
        neg_half_sentence2 = gen_neg_smaple(data)
        positive.append([" ".join(one[:pos_split]), " ".join(one[pos_split:]), "yes"])
        negative.append([neg_half_sentence1, neg_half_sentence2, "no"])
    print(f"正样本数{len(positive)}, 负样本数{len(negative)}")
    examples = positive + negative
    random.shuffle(examples)
    #拆分样本
    total = len(examples)
    train_num = int(total*0.8)
    test_num = int(total*0.1)
    train_data = examples[:train_num]
    test_data = examples[train_num:train_num+test_num]
    dev_data = examples[train_num+test_num:]
    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    with open(dev_file, 'w') as f:
        json.dump(dev_data, f)
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    print(f"训练集{train_num}, 测试集{test_num}, 开发集{total-train_num-test_num}")

if __name__ == '__main__':
    data = search_and_read()
    split_train_test(data)