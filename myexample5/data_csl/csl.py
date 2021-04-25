# coding=utf-8
# @Date  : 2021/4/27 10:30 上午
# @File  : custom_zh_en.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 中文文本摘要

from __future__ import absolute_import, division, print_function

import csv
import json
import os

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
中文文本摘要
训练集2800,开发集350, 测试集350
"""

_HOMEPAGE = "johnson homepage"

_LICENSE = "johnson license"

# 数据集下载地址
_URLs = {
    'csl': "https://huggingface.co/great-new-dataset-first_domain.zip",
}


#通常CamelCase命名
class CSLDataset(datasets.GeneratorBasedBuilder):
    """连贯性测试数据集"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="csl", version=VERSION, description="正常数量数据集"),
    ]

    DEFAULT_CONFIG_NAME = "custom_zh_en"
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "content": datasets.Value("string"),
                }
            ),
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
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": self.config.data_files['train'],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": self.config.data_files['test'],
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": self.config.data_files['validation'],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields 方法返回每个样本. """
        # 被函数_split_generators 调用，参数也是通过 gen_kwargs被传过来
        # 它负责打开给定的文件并从数据集中产生(key, example)元组
        # key是不重要的，只是习惯于这样
        with open(filepath, encoding="utf-8") as f:
            datas = json.load(f)
            for id_, data in enumerate(datas):
                yield id_, {
                    'title': data['title'],
                    'content': data['content'],
                    }