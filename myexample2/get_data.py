#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/1/14 10:35 上午
# @File  : get_data.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 获取数据，保存成txt


from read_hive import get_distinct_content

def get_txt(data, savefile='data/demo.txt'):
    """
    Args:
        data:
        savefile: 保存到txt，每行一个文本
    Returns:
    """
    data_list = data['content'].tolist()
    with open(savefile, 'w', encoding='utf-8') as file:
        for once in data_list:
            file.write(once + "\n")
    print(f'保存文件成功{savefile}')

if __name__ == '__main__':
    data = get_distinct_content(number=-1, ptime_keyword=None, not_cache=True, save_cache=False)
    get_txt(data=data, savefile='data/newbig.txt')