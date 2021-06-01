#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/5/17 2:45 下午
# @File  : data_split.txt.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 拆分数据集

import random
import os
random.seed(10)
import re

def split_data():
    # 5000条作为验证集，5000条作为测试集，剩下的都为训练集
    cn_files = ['chinese_67513.txt','final_cn.txt']
    en_files = ['english_67513.txt','final_en.txt']
    cn_lines = []
    en_lines = []
    for cn_file, en_file in zip(cn_files,en_files):
        with open(cn_file, 'r') as cf:
            lines = cf.readlines()
            cn_lines.extend(lines)
        with open(en_file, 'r') as ef:
            lines = ef.readlines()
            en_lines.extend(lines)
    print(f"共有数据 {len(cn_lines)}条")
    data_pair = []
    for cn,en in zip(cn_lines, en_lines):
        res = re.findall('[\u4e00-\u9fa5]+', en)
        if res:
            #跳过中文数据集
            continue
        data_pair.append([cn,en])
    print(f"过滤后的数据有 {len(data_pair)}条")
    #打乱数据
    # random.shuffle(data_pair)
    dev_data = data_pair[:2000]
    test_data = data_pair[2000:4000]
    train_data = data_pair[-10000:]
    #保存到文件
    parent_dir = '../'
    train_en_file = os.path.join(parent_dir, 'train.en')
    train_cn_file = os.path.join(parent_dir, 'train.cn')
    test_en_file = os.path.join(parent_dir, 'test.en')
    test_cn_file = os.path.join(parent_dir, 'test.cn')
    dev_en_file = os.path.join(parent_dir, 'dev.en')
    dev_cn_file = os.path.join(parent_dir, 'dev.cn')

    def write_pair_to_file(pair, cn_file, en_file):
        with open(cn_file, 'w') as cf, open(en_file, 'w') as ef:
            for d in pair:
                cn_text, en_text = d
                cf.write(cn_text)
                ef.write(en_text)
    write_pair_to_file(dev_data,dev_cn_file,dev_en_file)
    write_pair_to_file(train_data,train_cn_file,train_en_file)
    write_pair_to_file(test_data,test_cn_file,test_en_file)
    print(f"生成训练文件成功，保存到 {parent_dir}目录下")
if __name__ == '__main__':
    split_data()