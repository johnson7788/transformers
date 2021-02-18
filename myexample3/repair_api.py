#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/8 5:14 下午
# @File  : repair_api.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 加载模型，收到数据后进行预测



# coding=utf-8
# 文本序列分类微调脚本, 来自 examples/text-classification/run_glue.py

import logging
import os
import random
import sys
import re
from dataclasses import dataclass, field
from typing import Optional
from flask import Flask, request, jsonify, abort
import numpy as np
from datasets import load_dataset, load_metric, Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

app = Flask(__name__)
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    要输入哪些数据以输入我们的模型进行训练和评估的Arguments。
    使用`HfArgumentParser`，我们可以将此类转换为argparse参数，以便能够在命令行上指定它们。
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "训练任务的名称"},
    )
    task_dir: Optional[str] = field(
        default=None,
        metadata={"help": "训练数据文件夹路径"},
    )
    task_script: Optional[str] = field(
        default=None,
        metadata={"help": "训练的处理脚本位置"},
    )
    metric_script: Optional[str] = field(
        default=None,
        metadata={"help": "训练集的metric脚本的位置"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "分词后的最大总输入序列长度。 长度大于此长度的序列将被截断，较短的序列将被填充。"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖cached预处理数据集。"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "是否将所有样本填充到“max_seq_length”。 如果为False，则将动态填充样本到批次中的最大长度。"
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "包含训练数据的csv或json文件。 "}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "包含验证数据的csv或json文件。 "}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "包含测试数据的csv或json文件。"})


@dataclass
class ModelArguments:
    """
    关于我们要微调的model/config/tokenizer的参数。
    """

    model_name_or_path: str = field(
        default='repair/',
        metadata={"help": "huggingface.co/models中预训练模型或模型标识符的路径"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练的配置名称或路径（如果与model_name不同）"}
    )
    tokenizer_name: Optional[str] = field(
        default='bert-base-chinese', metadata={"help": "预训练的tokenizer生成器名称或路径（如果与model_name不同）"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "您想在哪里存储从s3下载的预训练模型,缓存模型文件夹"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速tokenizer之一(由tokenizer库支持)。"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "要使用的特定模型版本 (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@app.route("/api", methods=['POST'])
def api():
    """
    Args:
        最大序列长度是128，所以sentence长度最好不要超过120个字
        test_data: 需要预测的数据，是一个文字列表,[[sentence, [(english_word1,wrong_word1), (english_word2,wrong_word2)]],...]
    Returns:[[fix_sentence,[(english_word1,wrong_word1,predict_word), (english_word2,wrong_word2,predict_word)]],...]

    """
    jsonres = request.get_json()
    test_data = jsonres.get('data',None)
    # 按照wrong_word的长度排序
    sorted_data = []
    for data in test_data:
        words = sorted(data[1], key=lambda x:len(x[1]),reverse=True)
        sorted_data.append([data[0],words])
    print(f"收到数据{test_data}")
    results = do_predict(test_data)
    return jsonify(results)

def do_predict(test_data):
    """
    模型预测
    Args:
        test_data:
    Returns:

    """
    #设置日志格式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    #准备数据集, 和训练时的输入保持一致sentence1, sentence2
    test_dict = {'sentence1':[], 'sentence2':[]}
    for idx, data in enumerate(test_data):
        sentence1 = data[0]
        for eng_wrong in data[1]:
            eng_word, wrong_word = eng_wrong[0], eng_wrong[1]
            sentence2 = eng_word +'\n' + wrong_word + '\n'
            test_dict['sentence1'].append(sentence1)
            test_dict['sentence2'].append(sentence2)
    test_datasets = Dataset.from_dict(test_dict)

    sentence1_key, sentence2_key = "sentence1", "sentence2"

    # Set seed before initializing model.
    #随机数种子
    #num_labels 类别数量, output_mode  是任务类型，'classification'
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        """"
        # Tokenize the texts
        examples: {'idx': [0, 1], 'label': [5.0, 3.799999952316284], 'sentence1': ['A plane is taking off.', 'A man is playing a large flute.'], 'sentence2': ['An air plane is taking off.', 'A man is playing a flute.']}
        """
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=True, max_length=128, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    t_datasets = test_datasets.map(preprocess_function, batched=True, load_from_cache_file=False)

    logger.info("*** 预测 ***")
    predictions = trainer.predict(t_datasets)
    predictions = np.argmax(predictions.predictions, axis=1)

    # 是一个嵌套列表，子列表是label的名字, 去掉prediction的第一个和最后一个元素CLS, SEP
    predict_labels = [label_list[p] for p in predictions ]
    results = []
    #替换句子中的错误单词
    for idx,data in enumerate(test_data):
        sentence = data[0]
        correct_words = []
        for eng_wrong in data[1]:
            #开始替换
            eng_word, wrong_word = eng_wrong[0], eng_wrong[1]
            predict_word = predict_labels.pop(0)
            if predict_word == "DELETE":
                #如果预测是删除的关键字，那么就删掉这个词从原始句子中
                sentence = re.sub(wrong_word,'',sentence)
            else:
                sentence = re.sub(wrong_word,predict_word, sentence)
            eng_wrong.append(predict_word)
            correct_words.append(eng_wrong)
        results.append([sentence,correct_words])
    print(results)
    return results


def load_model():
    """
    加载模型，返回初始化后的模型
    Returns:
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    training_args = TrainingArguments(output_dir="output/repair", do_predict=True)
    logger.info(f"Training/evaluation parameters {training_args}")
    num_labels = len(label_list)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    #打印参数
    logger.info("Training/evaluation parameters %s", training_args)
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # .ckpt 是可以加载tensorflow的模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )
    return tokenizer, trainer

if __name__ == "__main__":
    import json
    labels_file = "dataset/repair/labels.json"
    with open(labels_file, 'r') as f:
        label_list = json.load(f)
    tokenizer, trainer = load_model()
    app.run(host='0.0.0.0', port=6666, debug=True, threaded=True)
