#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/4/25 10:45 上午
# @File  : t0.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import json
import random
import os

def split_data_dev(data, save_path, train_rate=0.8, dev_rate=0.1, test_rate=0.1):
    """
    保存为json文件，按比例保存
    train.json
    dev.json
    :param data: data 列表,
    :param weibodata: 如果weibodata存在，只作为训练集使用
    :return:
    """
    random.seed(30)
    random.shuffle(data)
    total = len(data)
    train_num = int(total * train_rate)
    dev_num = int(total * dev_rate)
    train_file = os.path.join(save_path, 'train.json')
    dev_file = os.path.join(save_path, 'dev.json')
    test_file = os.path.join(save_path, 'test.json')
    train_data = data[:train_num]
    dev_data = data[train_num:train_num+dev_num]
    test_data = data[train_num+dev_num:]
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f)
    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    print(f"保存成功，训练集{len(train_data)},开发集{len(dev_data)}, 测试集{len(test_data) }")
    return train_data, dev_data, test_data

if __name__ == '__main__':
    data_file = 'csl_data.json'
    with open(data_file, 'r') as f:
        all_data = json.load(f)
    split_data_dev(all_data, save_path='./')