# coding=utf-8

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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
    关于我们要微调或从头开始训练的model/config/tokenizer的参数。
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "权重初始化的模型checkpoint。如果要从头训练模型，则不要设置。 "
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "如果从头训练，传入一个模型的类型，例如deberta : " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练的config名称或路径(如果与model_name不同) "}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练的tokenizer名称或路径(如果与model_name不同) "}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "您想在哪里存储从huggingface.co下载的预训练模型 "},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用fast tokenizer, 需要tokenizer库支持，请查看官方文档 "},
    )


@dataclass
class DataTrainingArguments:
    """
    关于要输入哪些数据以供我们的模型进行训练和评估的参数。
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集的名称(通过datasets库)。 "}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "要使用的数据集的配置名称(通过datasets库)。 "}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "输入训练数据文件  (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "可选的输入评估数据文件，用于评估困惑度(a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的训练和评估集"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "分词后的最大总输入序列长度。 更长的序列将被截断。 "
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于预处理的进程数。 "},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "MLM模型的Mask token的比例，默认15%"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "数据集中文本的不同行是否将作为不同序列处理。"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "是否将所有样本填充到“max_seq_length”。 如果为False，将动态填充样本批量处理到批次中的最大长度。"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("需要数据集名称或训练/验证文件。")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # 在 src/transformers/training_args.py中查看所有可能的参数，或将--help标志传递给此脚本。
    # 现在，我们保留了不同的参数集，以使关注点更加清晰。
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们仅将一个参数传递给脚本，并且它是指向json文件的路径，那么让我们对其进行解析以获取参数。
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
            f"输出目录({training_args.output_dir}) 以及存在，并且不为空"
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # 记录每个进程的日志
    logger.warning(
        f"使用的 rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"是否分布式训练: {bool(training_args.local_rank != -1)}, 16-bits 半精度训练: {training_args.fp16}"
    )
    # 主进程的日志设为verbosity:
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("训练/评估参数 %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # 在分布式训练中，load_dataset函数可确保只有一个本地进程可以同时下载数据集。
    if data_args.dataset_name is not None:
        # 从hub下载和加载数据集。
        # 首先确定本地缓存了cache文件
        cache_script = os.path.join("data", data_args.dataset_name+".py")
        if not os.path.exists(cache_script):
            raise Exception("请检查本地是否存在相关脚本文件")
        datasets = load_dataset(cache_script, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 加载预训练模型和tokenizer
    #
    # Distributed training:
    # .from_pretrained方法可确保只有一本地个进程可以同时下载模型和vocab。
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("你正从头开始初始化一个新的config.")
    # tokenizer的设置
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
            "您可以用其它形式训练好之后，在这里使用，使用方法:  using --tokenizer_name."
        )
    #模型的设置
    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("从头开始训练一个模型")
        model = AutoModelForMaskedLM.from_config(config)
    #重设下tokenizer的大小，如果当我们从头训练新模型时，这是必须的
    model.resize_token_embeddings(len(tokenizer))

    # 处理数据集
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.line_by_line:
        # 按行处理， tokenize each nonempty line
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # 移除空行
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=data_args.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # 否则，我们将tokenize每个文本，然后将它们拼接在一起，然后再将它们分成较小的部分。
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
        #默认一次处理1000行
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"参数给定的 max_seq_length  ({data_args.max_seq_length}) 比模型的 ({tokenizer.model_max_length}) 最大长度长. 使用模型的最大长度 max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # 主要数据处理功能，可拼接数据集中的所有文本并生成max_seq_length的块。
        def group_texts(examples):
            # 拼接所有文本。
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # 我们删除一小部分，如果模型支持该字段，则可以添加padding，而不是删除，您可以根据需要自定义此部分。
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # 注意，使用batched=True`时，此映射一起处理1,000个文本，因此group_texts会丢弃这1,000个文本组中的每一个的余数。 您可以在此处调整该batch_size，但较高的值可能会较慢进行预处理。
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    # 这部分是随机mask token的设置
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

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
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** 开始评估 ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_mlm.txt")
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
