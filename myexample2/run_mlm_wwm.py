# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) with whole word masking on a
text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "初始化模型的checkpoint的权重文件，如果不设置，就相当于scratch，重新训练一个模型"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "如果重新训练一个scratch，必须传入一个模型的类型: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练模型的配置文件或路径，如果不和model_name一样的话"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练模型的tokenizer文件或路径，如果不和model_name一样的话"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "cache_dir， 下载huggingface.co模型的缓存目录"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用fast tokenizer"},
    )


@dataclass
class DataTrainingArguments:
    """
    关于要输入哪些数据以供我们的模型进行训练和评估的参数。
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "输入的训练数据文件(text文本文件)。 "})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "可选，评估数据文件，用于评估困惑度(文本文件)。"},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "用于中文全词masking的可选输入Train ref数据文件。"},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "用于中文全词masking的可选输入验证参考数据文件。 "},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的训练和评估集"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "tokenizer后的最大总输入序列长度。 更长的序列将被截断。 默认为模型的最大输入长度。 "
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于预处理的进程数。 "},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "MLM 语言模型的Mask token概率"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "是否将所有样本填充到“max_seq_length”。 如果为False，则在批次中的按照最大长度的样本动态填充。"
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` 仅支持 csv, a json or a txt file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` 仅支持 csv, a json or a txt file."


def add_chinese_references(dataset, ref_file):
    """
    加入chinese_ref这列，到数据集
    Args:
        dataset: 处理完成的数据集， 包括3列数据dict_keys(['attention_mask', 'input_ids', 'token_type_ids',])
        ref_file: 对应的中文全词的子词的引用文件

    Returns:
    dict_keys(['attention_mask', 'input_ids', 'token_type_ids', 'chinese_ref'])
    """
    with open(ref_file, "r", encoding="utf-8") as f:
        refs = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    assert len(dataset) == len(refs)

    dataset_dict = {c: dataset[c] for c in dataset.column_names}
    dataset_dict["chinese_ref"] = refs
    return Dataset.from_dict(dataset_dict)


def main():
    # 更多参数请查看  in src/transformers/training_args.py
    # 或通过--help查看
    # 现在，我们保留了不同的参数集，以使关注点更加清晰。

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们仅将一个参数传递给脚本，并且它是json文件的路径，
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # 将logger verbosity设置为 info(仅在main上):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # 在初始化模型之前设置种子。
    set_seed(training_args.seed)

    # 获取数据集：您可以提供自己的CSV / JSON / TXT训练和评估文件(请参见下文)
    # 或仅提供位于hub的可用公共数据集之一的名称，网址为  https://huggingface.co/datasets/
    # (该数据集将自动从数据集中心下载)。
    #
    # 对于CSV / JSON文件，此脚本将使用名为'text'的列或如果未找到名为'text'的列的第一列。
    # 您可以轻松调整此行为(请参见下文)。
    #
    # 在分布式训练中，load_dataset函数可确保只有一个local进程可以同时下载数据集。
    #
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    # datasets = load_dataset(path=extension, data_files=data_files)
    datasets = load_dataset(path="data/text.py", data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 加载 pretrained model and tokenizer
    #
    # 分布式训练 training:
    # .from_pretrained方法可确保只有一个local进程可以同时下载模型和vocab
    # 加载模型的配置
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    #加载tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "您正在从头实例化一个新的tokenizer。 此脚本不支持此功能。 "
            "您可以使用其它脚本进行--tokenizer_name保存，并从此处使用和加载。 "
        )
    #加载模型，或者随机初始化模型
    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("从头scratch开始训练一个新的模型")
        model = AutoModelForMaskedLM.from_config(config)
    #可以根据字典重新调整嵌入层的大小, 如果单词表大小不变，embedding也不变, [old_vocab_size, 768] ---> [new_vocab_size, 768]
    model.resize_token_embeddings(len(tokenizer))
    # 预处理数据集
    # 首先，我们所有文本分词。
    if training_args.do_train:
        # eg: column_names: ['text']
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    # eg text_column_name: text
    text_column_name = "text" if "text" in column_names else column_names[0]
    # padding的方式
    padding = "max_length" if data_args.pad_to_max_length else False
    def tokenize_function(examples):
        """
        处理2条数据的数据
        Args:
            examples: {'text': ['古龙洗发水，洗完头发不干燥、也不容易油、不痒，味道持久，非常柔顺，而且泡泡很容易冲洗干净泡沫非常细腻，洗后头发很滑很顺，洗了之后就头发很蓬松，很香，而且我洗了是没有头皮屑的', '老用户了，一直在用满婷，感觉对控痘控油效果挺好的']}
        Returns:
            返回包括一个batch的数据的， dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        """
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        newexample = tokenizer(examples["text"], padding=padding, truncation=True, max_length=data_args.max_seq_length)
        return newexample
    #处理数据
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # 添加中文引用(如果提供的话) ，训练集的全词引用
    if data_args.train_ref_file is not None:
        tokenized_datasets["train"] = add_chinese_references(tokenized_datasets["train"], data_args.train_ref_file)
    #验证集的全词引用
    if data_args.validation_ref_file is not None:
        tokenized_datasets["validation"] = add_chinese_references(
            tokenized_datasets["validation"], data_args.validation_ref_file
        )

    # Data collator, 输入模型前的数据处理器
    # 这将进行随机masked 全词的token，具体mask全词的方式，请进入函数查看
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        logger.info("*** 开始训练模型 ***")
        #model_path加载模型的optimizer/scheduler路径， 需要有optimizer.pt和scheduler.pt文件在路径下，否则会初始化
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        trainer.train(model_path=model_path)
        #保存训练好的模型到output目录
        trainer.save_model(output_dir=training_args.output_dir)  # Saves the tokenizer too for easy upload

    # 评估
    results = {}
    if training_args.do_eval:
        logger.info("*** 开始评估模型 ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm_wwm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
