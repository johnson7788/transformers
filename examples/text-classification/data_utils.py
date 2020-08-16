import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from transformers.tokenization_bart import BartTokenizer, BartTokenizerFast
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
from transformers.data.processors.utils import InputFeatures, DataProcessor, InputExample
from transformers.file_utils import is_tf_available
from docx2python import docx2python
from tqdm import tqdm
import subprocess

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    将数据文件加载到``InputFeatures''列表中

    Args:
        examples: 包含样本的“ InputExamples”或“ tf.data.Dataset”的列表。
        tokenizer: 将tokenize样本的tokenizer的实例
        max_length: 最大示例长度。 默认为tokenizer's 的最大长度
        task: GLUE task
        label_list: 标签列表。 可以使用processor.get_labels（）方法从处理器获取
        output_mode: 指示输出模式的字符串  Either ``regression`` or ``classification``

    Returns:
        如果examples输入是tf.data.Dataset，则将返回tf.data.Dataset，其中包含特定于任务的功能。
         如果输入是“ InputExamples”的列表，则将返回任务特定的“ InputFeatures”列表，可以将其输入模型。

    """
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    #label 字符串到id的映射表
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)
    #获取所有样本的labels
    labels = [label_from_example(example) for example in examples]
    #所有样本字符到id的，padding或traucate 后的结果
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    #把input_ids, attention_mask, token_type_ids, label 放到一个对象InputFeatures 里面
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        # 把input_ids, attention_mask, token_type_ids, label 放到一个对象InputFeatures 里面
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)
    #打印前5个样本
    logger.info("*** 打印前5个样本 ***")
    for i, example in enumerate(examples[:5]):
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class WenbenProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "train"), "train")

    def get_dev_examples(self, data_dir):
        """eval文件夹是dev评估数据"""
        return self._create_examples(os.path.join(data_dir, "eval"), "dev")

    def get_test_examples(self, data_dir):
        """predict 目录是test数据."""
        return self._create_examples(os.path.join(data_dir, "predict"), "test")

    def get_labels(self):
        """See base class."""
        return ['组织机构代码证', '营业执照', '身份证', '事业单位法人证书', '学位证', '其它', '四六级', '环境证书', '驾照', '毕业证']

    def _create_examples(self, path, set_type):
        """
        创建数据集
        Args:
            path: train, dev, test数据集路径
            set_type: 标记数据类型, train, dev, test
        Returns:
        """
        examples = []
        #样本计数
        count = 0
        dirs = os.listdir(path)
        #如果是测试数据，或者预测数据，不是双层文件夹，是一层文件夹
        if set_type != 'test':
            for dir in dirs:
                files = os.listdir(os.path.join(path, dir))
                for file in tqdm(files):
                    file_content = self.docx2text(os.path.join(path, dir, file))
                    # 过滤掉内容少于5个字符的无意义文档
                    if len(file_content) > 5:
                        guid = "%s-%s" % (set_type, count)
                        count += 1
                        text_a = file_content
                        label = dir
                        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        else:
            #设置label为None
            for file in dirs:
                file_content = self.docx2text(os.path.join(path, file))
                guid = "%s-%s" % (set_type, count)
                count += 1
                text_a = file_content
                label = None
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def docx2text(self,filename):
        """
        :param filename: docx源文件
        :return: docx的文字内容
        """

        def flatten(S):
            """
            展平嵌套列表
            :param S: 嵌套列表
            :return: 单个不嵌套的列表
            """
            if S == []:
                return S
            if isinstance(S[0], list):
                return flatten(S[0]) + flatten(S[1:])
            return S[:1] + flatten(S[1:])

        if filename.split('.')[-1] == "docx":
            # 提取文本
            doc_result = docx2python(filename)
            # 展开结果
            res = flatten(doc_result.body)
            # 去除空格
            res = [r for r in res if r.strip()]
            # 返回成原来格式
            content = '。'.join(res)
        elif filename.split('.')[-1] == "doc":
            content = subprocess.check_output(['antiword', filename])
            content = content.decode('utf-8')
        return content

glue_processors = {
    "sst-2": Sst2Processor,
    "wenben": WenbenProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "wenben": "classification",
}

glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "wenben":10,
}

@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "要训练的任务的名称: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "数据目录，目录下是.tsv文件或者其他数据文件"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "tokenization之后的序列最大长度，超过则被截断，过短则被padded"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖训练和eval的数据集cached文件"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GlueDataset(Dataset):
    """
    这很快将被与框架无关的方法所取代

    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            args: 数据集的参数
            tokenizer: 使用的tokenizer
            limit_length: 设置取多少条数据样本
            mode: 是train，还是dev，还是text的数据集
            cache_dir: 使用的数据的cache目录是
        """
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        #生成cached文件的名字
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        #获取label的列表
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # 确保只有分布式训练中的第一个流程会处理数据集，其他流程将使用缓存。
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"直接从cached file中加载features  {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"从数据文件中创建features {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # 保存cache文件
                logger.info(
                    "保存 features文件到cached file %s [花费秒数 %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(
            labels
        ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"mnli/acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"mnli-mm/acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(
            labels
        ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)


