#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/1/27 10:30 上午
# @File  : gen_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 生成二分类的数据集, 上下句是连贯的，那么值为1，否则为0
from pathlib import Path
import re
def search_and_read(path='/opt/nlp/pycorrect'):
    data = []
    for fpath in Path(path).rglob('*.mix'):
        with open(fpath) as f:
            for line in f:
                #匹配英文，并且，单词数大于10
                if len(line.split()) > 10 and not re.search("[\u4e00-\u9fa5]", line):
                    data.append(line)
    return data