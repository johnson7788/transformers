# MSRA中文NER
# 支持参数 -h
```
usage: cosmetic_run_ner.py [-h] --model_name_or_path MODEL_NAME_OR_PATH
                           [--config_name CONFIG_NAME]
                           [--tokenizer_name TOKENIZER_NAME]
                           [--cache_dir CACHE_DIR] [--task_name TASK_NAME]
                           [--dataset_name DATASET_NAME]
                           [--script_file SCRIPT_FILE]
                           [--dataset_config_name DATASET_CONFIG_NAME]
                           [--train_file TRAIN_FILE]
                           [--validation_file VALIDATION_FILE]
                           [--test_file TEST_FILE] [--overwrite_cache]
                           [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]
                           [--pad_to_max_length] [--max_length MAX_LENGTH]
                           [--label_all_tokens] --output_dir OUTPUT_DIR
                           [--overwrite_output_dir] [--do_train] [--do_eval]
                           [--do_predict] [--model_parallel]
                           [--evaluation_strategy {EvaluationStrategy.NO,EvaluationStrategy.STEPS,EvaluationStrategy.EPOCH}]
                           [--prediction_loss_only]
                           [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                           [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                           [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                           [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                           [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                           [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS]
                           [--learning_rate LEARNING_RATE]
                           [--weight_decay WEIGHT_DECAY]
                           [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                           [--adam_epsilon ADAM_EPSILON]
                           [--max_grad_norm MAX_GRAD_NORM]
                           [--num_train_epochs NUM_TRAIN_EPOCHS]
                           [--max_steps MAX_STEPS]
                           [--warmup_steps WARMUP_STEPS]
                           [--logging_dir LOGGING_DIR] [--logging_first_step]
                           [--logging_steps LOGGING_STEPS]
                           [--save_steps SAVE_STEPS]
                           [--save_total_limit SAVE_TOTAL_LIMIT] [--no_cuda]
                           [--seed SEED] [--fp16]
                           [--fp16_opt_level FP16_OPT_LEVEL]
                           [--local_rank LOCAL_RANK]
                           [--tpu_num_cores TPU_NUM_CORES]
                           [--tpu_metrics_debug] [--debug]
                           [--dataloader_drop_last] [--eval_steps EVAL_STEPS]
                           [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                           [--past_index PAST_INDEX] [--run_name RUN_NAME]
                           [--disable_tqdm DISABLE_TQDM]
                           [--no_remove_unused_columns]
                           [--label_names LABEL_NAMES [LABEL_NAMES ...]]
                           [--load_best_model_at_end]
                           [--metric_for_best_model METRIC_FOR_BEST_MODEL]
                           [--greater_is_better GREATER_IS_BETTER]
                           [--ignore_data_skip]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from
                        huggingface.co/models
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as
                        model_name
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as
                        model_name
  --cache_dir CACHE_DIR
                        Where do you want to store the pretrained models
                        downloaded from huggingface.co
  --task_name TASK_NAME
                        The name of the task (ner, pos...).
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets
                        library).
  --script_file SCRIPT_FILE
                        处理数据集的文件路径
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the
                        datasets library).
  --train_file TRAIN_FILE
                        The input training data file (a csv or JSON file).
  --validation_file VALIDATION_FILE
                        An optional input evaluation data file to evaluate on
                        (a csv or JSON file).
  --test_file TEST_FILE
                        An optional input test data file to predict on (a csv
                        or JSON file).
  --overwrite_cache     Overwrite the cached training and evaluation sets
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing.
  --pad_to_max_length   Whether to pad all samples to model maximum sentence
                        length. If False, will pad the samples dynamically
                        when batching to the maximum length in the batch. More
                        efficient on GPU but very bad for TPU.
  --max_length MAX_LENGTH
                        padding的最大序列长度，默认是64，如果是bert，最长是512
  --label_all_tokens    Whether to put the label for one word on all tokens of
                        generated by that word or just on the one (in which
                        case the other tokens will have a padding index).
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --overwrite_output_dir
                        Overwrite the content of the output directory.Use this
                        to continue training if output_dir points to a
                        checkpoint directory.
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the dev set.
  --do_predict          Whether to run predictions on the test set.
  --model_parallel      If there are more than one devices, whether to use
                        model parallelism to distribute the model's modules
                        across devices.
  --evaluation_strategy {EvaluationStrategy.NO,EvaluationStrategy.STEPS,EvaluationStrategy.EPOCH}
                        Run evaluation during training at each logging step.
  --prediction_loss_only
                        When performing evaluation and predictions, only
                        returns the loss.
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for training.
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for evaluation.
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size`
                        is preferred. Batch size per GPU/TPU core/CPU for
                        training.
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size`
                        is preferred.Batch size per GPU/TPU core/CPU for
                        evaluation.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before
                        performing a backward/update pass.
  --eval_accumulation_steps EVAL_ACCUMULATION_STEPS
                        Number of predictions steps to accumulate before
                        moving the tensors to the CPU.
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --weight_decay WEIGHT_DECAY
                        Weight decay if we apply some.
  --adam_beta1 ADAM_BETA1
                        Beta1 for Adam optimizer
  --adam_beta2 ADAM_BETA2
                        Beta2 for Adam optimizer
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_dir LOGGING_DIR
                        Tensorboard log dir.
  --logging_first_step  Log the first global_step
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints.Deletes the
                        older checkpoints in the output_dir. Default is
                        unlimited checkpoints
  --no_cuda             Do not use CUDA even when it is available
  --seed SEED           random seed for initialization
  --fp16                Whether to use 16-bit (mixed) precision (through
                        NVIDIA apex) instead of 32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --tpu_num_cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script)
  --tpu_metrics_debug   Deprecated, the use of `--debug` is preferred. TPU:
                        Whether to print debug metrics
  --debug               Whether to print debug metrics on TPU
  --dataloader_drop_last
                        Drop the last incomplete batch if it is not divisible
                        by the batch size.
  --eval_steps EVAL_STEPS
                        Run an evaluation every X steps.
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading
                        (PyTorch only). 0 means that the data will be loaded
                        in the main process.
  --past_index PAST_INDEX
                        If >=0, uses the corresponding part of the output as
                        the past state for next step.
  --run_name RUN_NAME   An optional descriptor for the run. Notably used for
                        wandb logging.
  --disable_tqdm DISABLE_TQDM
                        Whether or not to disable the tqdm progress bars.
  --no_remove_unused_columns
                        Remove columns not required by the model when using an
                        nlp.Dataset.
  --label_names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that
                        correspond to the labels.
  --load_best_model_at_end
                        Whether or not to load the best model found during
                        training at the end of training.
  --metric_for_best_model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models.
  --greater_is_better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be
                        maximized or not.
  --ignore_data_skip    When resuming training, whether or not to skip the
                        first epochs and batches to get to the same training
                        data.
```

# 测试英文Conll2003的效果， 默认3个epoch
```buildoutcfg
python conll2003_run_ner.py --model_name_or_path bert-base-uncased --dataset_name conll2003 --output_dir conll-ner --do_train --do_eval --save_total_limit 5
12/09/2020 08:33:19 - INFO - __main__ -   ***** Eval results *****
12/09/2020 08:33:19 - INFO - __main__ -     eval_loss = 0.050854604691267014
12/09/2020 08:33:19 - INFO - __main__ -     eval_accuracy_score = 0.9894863907168724
12/09/2020 08:33:19 - INFO - __main__ -     eval_precision = 0.9440360841964585
12/09/2020 08:33:19 - INFO - __main__ -     eval_recall = 0.9510265903736116
12/09/2020 08:33:19 - INFO - __main__ -     eval_f1 = 0.9475184439973172
12/09/2020 08:33:19 - INFO - __main__ -     epoch = 3.0
```

# 使用BERT的效果,最大序列长度128， 默认3个epoch
```buildoutcfg
cd myexample
python msra_run_ner.py --model_name_or_path bert-base-chinese --dataset_name msra_ner --output_dir msra_ner --do_train --do_eval --max_length 128 --save_total_limit 5
12/10/2020 06:11:51 - INFO - __main__ -   ***** Eval results *****
12/10/2020 06:11:51 - INFO - __main__ -     eval_loss = 0.03320741653442383
12/10/2020 06:11:51 - INFO - __main__ -     eval_accuracy_score = 0.9937063237924316
12/10/2020 06:11:51 - INFO - __main__ -     eval_precision = 0.9424219910846954
12/10/2020 06:11:51 - INFO - __main__ -     eval_recall = 0.9494760479041916
12/10/2020 06:11:51 - INFO - __main__ -     eval_f1 = 0.9459358687546606
12/10/2020 06:11:51 - INFO - __main__ -     epoch = 3.0
```

# 使用MacBERT的效果,最大序列长度128, 默认3个epoch
```buildoutcfg
cd myexample
python msra_run_ner.py --model_name_or_path hfl/chinese-macbert-base --dataset_name msra_ner --output_dir msra_ner --do_train --do_eval --max_length 128 --save_total_limit 5
12/09/2020 09:53:40 - INFO - __main__ -   ***** Eval results *****
12/09/2020 09:53:40 - INFO - __main__ -     eval_loss = 0.03303384408354759
12/09/2020 09:53:40 - INFO - __main__ -     eval_accuracy_score = 0.9935772542608311
12/09/2020 09:53:40 - INFO - __main__ -     eval_precision = 0.9512422940407248
12/09/2020 09:53:40 - INFO - __main__ -     eval_recall = 0.9528443113772455
12/09/2020 09:53:40 - INFO - __main__ -     eval_f1 = 0.9520426287744227
12/09/2020 09:53:40 - INFO - __main__ -     epoch = 3.0
```

# cosemtics MacBert
使用自定义数据集:包含的column  ['id', 'ner_tags', 'tokens']
```buildoutcfg
cd myexample
python cosmetics_run_ner.py --model_name_or_path hfl/chinese-macbert-base --dataset_name cosmetic_ner --script_file data/cosmetic_ner.py --train_file dataset/cosmetic/train.txt --validation_file dataset/cosmetic/dev.txt --test_file dataset/cosmetic/test.txt --output_dir cosmetic_ner --do_train --do_eval --max_length 64 --save_total_limit 5
```
