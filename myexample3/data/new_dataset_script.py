# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os

import datasets



_CITATION = """\
@InProceedings{huggingface:dataset,
title = {smooth test},
authors={johnson
},
year={2020}
}
"""

#数据集描述
_DESCRIPTION = """\
连贯性测试数据集
"""

_HOMEPAGE = "johnson homepage"

_LICENSE = "johnson license"

# 数据集下载地址
_URLs = {
    'mini_smooth': "https://huggingface.co/great-new-dataset-first_domain.zip",
    'std_smooth': "https://huggingface.co/great-new-dataset-second_domain.zip",
}


#通常CamelCase命名
class SmoothDataset(datasets.GeneratorBasedBuilder):
    """连贯性测试数据集"""

    VERSION = datasets.Version("1.1.0")

    # 可以使用如下方式加载，
    # data = datasets.load_dataset(path='my_dataset', name='mini_smooth')
    # data = datasets.load_dataset(path='my_dataset', name='std_smooth')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="mini_smooth", version=VERSION, description="mini数据集"),
        datasets.BuilderConfig(name="std_smooth", version=VERSION, description="正常数量数据集"),
    ]

    DEFAULT_CONFIG_NAME = "std_smooth"

    def _info(self):
        # 指定datasets.DatasetInfo类包含的数据集信息
        # 判断传入的参数，  data = datasets.load_dataset(path='my_dataset', name='std_smooth')
        if self.config.name == "std_smooth":
            features = datasets.Features(
                {
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                    "label": datasets.Value("string")
                    # 还可传入其它特征
                }
            )
        else:  # 这里假设它们传入的features一样，其实可以根据name进行修改
            features = datasets.Features(
                {
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                    "label": datasets.Value("string")
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
        """下载数据集"""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive 
        my_urls = _URLs[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # 下面的参数将传给 _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        # TODO: This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "first_domain":
                    yield id_, {
                        "sentence": data["sentence"],
                        "option1": data["option1"],
                        "answer": "" if split == "test" else data["answer"],
                    }
                else:
                    yield id_, {
                        "sentence": data["sentence"],
                        "option2": data["option2"],
                        "second_domain_answer": "" if split == "test" else data["second_domain_answer"],
                    }
