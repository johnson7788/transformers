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

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for id_, row in enumerate(data):
                yield id_, {
                    "sentence1": row[1].strip(),   #句子
                    # "sentence2": row[0] + row[2],  #英语单词+错误单词
                    "sentence2": row[2].strip(),  #英语单词+错误单词
                    "label": row[3].strip(),  #正确单词
                }