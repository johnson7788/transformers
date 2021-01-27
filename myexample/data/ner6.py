# coding=utf-8

"""分为5个维度和其它维度,     tags = {"成分":"COM",
            "功效":"EFF",
            "香味":"FRA",
            "包装":"PAC",
            "使用感受":"SKI",  #使用感受即"肤感"
            "其它":"O"} """

import logging

import datasets


_CITATION = """\
@inproceedings{test,
  author    = {johnson},
  title     = {Cosmetic dataset},
  booktitle = {test},
  pages     = {100},
  publisher = {test},
  year      = {2020}
}
"""

_DESCRIPTION = """\
ner识别数据
"""

class CosmeticNerConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """构建配置文件

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CosmeticNerConfig, self).__init__(**kwargs)


class CosmeticNer(datasets.GeneratorBasedBuilder):
    """Cosmetic NER dataset."""
    # 注意这里的name会与load_dataset加载时的name进行对应,否则会提示找不到
    BUILDER_CONFIGS = [
        CosmeticNerConfig(name="ner6", version=datasets.Version("1.0.0"), description="NER 6 dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-COM",
                                "I-COM",
                                "B-EFF",
                                "I-EFF",
                                "B-FRA",
                                "I-FRA",
                                "B-PAC",
                                "I-PAC",
                                "B-SKI",
                                "I-SKI",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://example.com/Cosmetic",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
                返回SplitGenerators生成器, 下载数据和定义拆分数据
        可以使用self的config，里面包含了data_dir, data_files的从load_dataset传过来的参数
        例如: dataset = load_dataset('msra/msra_ner.py', data_dir='msra', data_files={'train': 'msra/mini.txt', 'test': 'msra/mini.txt'})
        那么self中会有如下参数
         config = {BuilderConfig} BuilderConfig(name='default-21c5277b4d9558dd', version=0.0.0, data_dir='msra', data_files={'train': 'msra/mini.txt', 'test': 'msra/mini.txt'}, description=None)
         data_dir = {str} 'msra'
         data_files = {dict: 2} {'train': 'msra/mini.txt', 'test': 'msra/mini.txt'}
         description = {NoneType} None
         name = {str} 'default-21c5277b4d9558dd'
        Args:
            dl_manager: dl_manager is a nlp.download.DownloadManager 下载器，从url下载数据并解压

        Returns:

        """
        if not self.config.data_files:
            print("错误，必须提供data_files参数")
            return None
        #注意gen_kwargs
        #调用_generate_examples加载训练集样本, 这个gen_kwargs是传给_generate_examples的，可以自己设定
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": self.config.data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": self.config.data_files["test"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": self.config.data_files["validation"]}),
        ]

    def _generate_examples(self, filepath):
        """
        yields 返回样本, 参数由_split_generators返回的gen_kwargs提供
        :param filepath: 读取的文件路径
        :return: Yiled 一个 (key, example) tuples的元祖数据
        Args:
            filepath:
        Returns:

        """
        logging.info("开始生成样本= %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                line_stripped = line.strip()
                if line_stripped == "":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    splits = line_stripped.split("\t")
                    if len(splits) == 1:
                        splits.append("O")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1])
            #最后一条样本
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
