# coding=utf-8
# @Date  : 2021/1/27 10:30 上午
# @File  : gen_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试句子中的单词的正确的翻译，直接给出正确的翻译的单词

from __future__ import absolute_import, division, print_function

import csv
import json
import os
import re

import datasets



_CITATION = """\
@InProceedings{huggingface:dataset,
title = {repair test},
authors={johnson
},
year={2020}
}
"""

#数据集描述
_DESCRIPTION = """\
翻译的句子的数据集
"""

_HOMEPAGE = "johnson homepage"

_LICENSE = "johnson license"

# 数据集下载地址
_URLs = {
    'repair': "https://huggingface.co/great-new-dataset-first_domain.zip",
}

#加载标签
import json
labels_file = "dataset/repair/labels.json"
with open(labels_file, 'r') as f:
    LABELS = json.load(f)

#通常CamelCase命名
class RepairDataset(datasets.GeneratorBasedBuilder):
    """连贯性测试数据集"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="repair", version=VERSION, description="正常数量数据集"),
    ]

    DEFAULT_CONFIG_NAME = "repair"

    def _info(self):
        # 指定datasets.DatasetInfo类包含的数据集信息
        features = datasets.Features(
            {
                "sentence1": datasets.Value("string"),
                "sentence2": datasets.Value("string"),
                "label": datasets.features.ClassLabel(names=LABELS)
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            #不同的数据集可以不同的特征，即不同的column
            features=features,
            # 如果特征包含一个通用的(input, target)元组，请在此处指定它们。They'll be used in builder.as_dataset,  as_supervised=True
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """下载数据集
        此方法的任务是下载/提取数据并根据配置定义拆分
        根据不同的配置BUILDER_CONFIGS，和数据集的name定义
        """
        # dl_manager是一个datasets.download.DownloadManager，可用于下载和提取URL，
        # 它可以接受任何类型或嵌套的列表/字典，并将返回相同的结构，url也可以替换为局部文件的路径。
        # 默认情况下，将提取压缩包，如果文件是压缩的，并返回提取压缩的缓存文件夹的路径，而不是压缩文件
        if self.config.data_dir:
            data_dir = self.config.data_dir
        else:
            my_urls = _URLs[self.config.name]
            data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.json"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.json"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields 方法返回每个样本. """
        # 被函数_split_generators 调用，参数也是通过 gen_kwargs被传过来
        # 它负责打开给定的文件并从数据集中产生(key, example)元组
        # key是不重要的，只是习惯于这样
        max_seq_length = 70
        SPECIAL = '_'
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for id_, row in enumerate(data):
                row = [r.strip() for r in row]
                engword, texta, keyword, label, blank = row
                # 计算下texta的真正的应该保留的长度，-3是减去CLS,SEP,SEP，这3个special的token
                max_texta_length = max_seq_length - len(keyword) - 3
                # 我们尝试在原句子中MASK掉关键字,但如果句子长度大于我们模型中的最大长度，我们需要截断
                iter = re.finditer(keyword, texta)
                for m in iter:
                    start_idx, end_idx = m.span()
                    texta_list = list(texta)
                    texta_list.insert(start_idx, SPECIAL)
                    texta_list.insert(end_idx + 1, SPECIAL)
                    texta_special = ''.join(texta_list)
                    special_keyword = SPECIAL + keyword + SPECIAL
                    # 开始检查长度, 如果长度大于最大序列长度，我们要截取关键字上下的句子，直到满足最大长度以内,截取时用句子分隔的方式截取
                    if len(texta_special) > max_texta_length:
                        # 需要对texta进行阶段,采取怎样的截断方式更合适,按逗号和句号和冒号分隔
                        texta_split = re.split('[，。：]', texta_special)
                        # 确定keyword在列表中的第几个元素中
                        special_keyword_idx = 0
                        for t_idx, t in enumerate(texta_split):
                            if special_keyword in t:
                                special_keyword_idx = t_idx
                        # 先从距离special_keyword_idx最远的地方的句子开始去掉，直到满足序列长度小于max_seq_length, 也要考虑添加逗号后的长度，所以要减去元素个数max_texta_length-len(texta_split)+1
                        while len(texta_split) > 1 and sum(len(t) for t in texta_split) > (
                                max_texta_length - len(texta_split) + 1):
                            # 选择从列表的左面弹出句子还是，右面弹出句子, 极端情况是只有2个句子，special_keyword_idx在第一个句子中，那么应该弹出第二个句子
                            if len(texta_split) / 2 - special_keyword_idx >= 0:
                                # special_keyword_idx在列表的左半部分，应该从后面弹出句子
                                texta_split.pop()
                            else:
                                # 从列表的开头弹出句子
                                texta_split.pop(0)
                        # 如果仅剩一个句子了，长度仍然大于最大长度, 那么只好强制截断了, 如果关键字中有，。：,那么也是有问题的，只好强制截断
                        key_symbol = False
                        for symbol in ['，', '。', '：']:
                            if symbol in keyword:
                                key_symbol = True
                        if (len(texta_split) == 1 and len(texta_split[0]) > max_texta_length) or key_symbol:
                            left_text = texta_special[:start_idx]
                            right_text = texta_special[end_idx + 2:]
                            # 如果左侧长度大于max_texta_length的一半，那么截断
                            keep_length = int((max_texta_length - len(keyword)) / 2)
                            if len(left_text) > keep_length:
                                left_text = left_text[-keep_length:]
                            if len(right_text) > keep_length:
                                right_text = right_text[:keep_length]
                            text_a = left_text + special_keyword + right_text
                        else:
                            text_a = '，'.join(texta_split)
                    else:
                        text_a = texta_special
                    yield id_, {
                        "sentence1": text_a,   #句子
                        # "sentence2": row[0] + row[2],  #英语单词+错误单词
                        "sentence2": keyword,  #英语单词+错误单词
                        "label": label,  #正确单词
                    }