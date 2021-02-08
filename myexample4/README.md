# DeBERTa 模型测试
```
命令: 
python sequence_classfication.py --model_name_or_path microsoft/deberta-base --task_name sst2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output/sst2/  --evaluation_strategy steps --eval_steps 500

epoch = 3.0
eval_accuracy = 0.9541284403669725
eval_loss = 0.21425390243530273
eval_runtime = 8.2135
eval_samples_per_second = 106.166

Electra base
python sequence_classfication.py --model_name_or_path google/electra-base-discriminator --task_name sst2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output/sst2/  --evaluation_strategy steps --eval_steps 500

02/04/2021 05:22:46 - INFO - __main__ -   ***** Eval results sst2 *****
02/04/2021 05:22:46 - INFO - __main__ -     epoch = 3.0
02/04/2021 05:22:46 - INFO - __main__ -     eval_accuracy = 0.9472477064220184
02/04/2021 05:22:46 - INFO - __main__ -     eval_loss = 0.19900847971439362
02/04/2021 05:22:46 - INFO - __main__ -     eval_runtime = 7.2291
02/04/2021 05:22:46 - INFO - __main__ -     eval_samples_per_second = 120.623

#加上半精度
python sequence_classfication.py \
  --model_name_or_path microsoft/deberta-base \
  --task_name sst2 \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir output/sst2/ \
  --fp16
```

# 继续训练DeBERTa
```
python run_mlm.py --model_name_or_path microsoft/deberta-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir output/mydeberta --per_device_train_batch_size 24 --gradient_accumulation_steps 2 --max_seq_length 128
python run_mlm.py \
    --model_name_or_path microsoft/deberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir output/mydeberta
    
# 输出结果:
perplexity = 10.445827453447645
```

## 使用继续训练好的DeBERTa微调SST2数据
模型保存的目录 output/mydeberta
python sequence_classfication.py --model_name_or_path output/mydeberta --task_name sst2 --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir output/sst2/  --evaluation_strategy steps --eval_steps 500

# 使用自定义的中文训练集继续预训练模型, 示例 bert-base-chinese, 2.75GB, 样本行数796万
python run_mlm.py --model_name_or_path bert-base-chinese --dataset_name demo --dataset_config_name demo --data_dir dataset/demo --do_train --do_eval --output_dir output/mybert --per_device_train_batch_size 24 --gradient_accumulation_steps 2 --max_seq_length 128
```buildoutcfg
生成缓存数据集大概64GB
du -sh /root/.cache/huggingface/datasets/*
64G	/root/.cache/huggingface/datasets/demo_dataset

使用CPU训练是不切实际的，如下所示:
 0% 0/10778 [00:00<?, ?ba/s][WARNING|tokenization_utils_base.py:3214] 2021-02-08 01:54:54,708 >> Token indices sequence length is longer than the specified maximum sequence length for this model (912 > 512). Running this sequence through the model will result in indexing errors
100% 10778/10778 [37:09<00:00,  4.84ba/s]
100% 1/1 [00:00<00:00, 18.74ba/s]
100% 1/1 [00:00<00:00, 32.21ba/s]
100% 10778/10778 [3:51:16<00:00,  1.29s/ba]
100% 1/1 [00:00<00:00, 11.34ba/s]
100% 1/1 [00:00<00:00, 22.13ba/s]
[INFO|trainer.py:429] 2021-02-08 06:43:26,808 >> The following columns in the training set don't have a corresponding argument in `BertForMaskedLM.forward` and have been ignored: special_tokens_mask.
[INFO|trainer.py:429] 2021-02-08 06:43:26,813 >> The following columns in the evaluation set don't have a corresponding argument in `BertForMaskedLM.forward` and have been ignored: special_tokens_mask.
/content/transformers/src/transformers/trainer.py:702: FutureWarning: `model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` instead.
  FutureWarning,
[INFO|trainer.py:832] 2021-02-08 06:43:26,822 >> ***** Running training *****
[INFO|trainer.py:833] 2021-02-08 06:43:26,822 >>   Num examples = 7961583
[INFO|trainer.py:834] 2021-02-08 06:43:26,822 >>   Num Epochs = 3
[INFO|trainer.py:835] 2021-02-08 06:43:26,822 >>   Instantaneous batch size per device = 24
[INFO|trainer.py:836] 2021-02-08 06:43:26,822 >>   Total train batch size (w. parallel, distributed & accumulation) = 48
[INFO|trainer.py:837] 2021-02-08 06:43:26,822 >>   Gradient Accumulation steps = 2
[INFO|trainer.py:838] 2021-02-08 06:43:26,822 >>   Total optimization steps = 497598
  0% 56/497598 [1:06:43<9887:09:08, 71.54s/it]
```
# 使用自定义的中文训练集继续预训练模型, 示例 clue/roberta_chinese_base , 2.75GB , 注意使用bert的tokenizer
python run_mlm.py --model_name_or_path clue/roberta_chinese_base --tokenizer_name bert-base-chinese --dataset_name demo --dataset_config_name demo --data_dir dataset/demo --do_train --do_eval --output_dir output/mybert --per_device_train_batch_size 4 --gradient_accumulation_steps 12 --max_seq_length 512

# 分布式训练
## 2个节点的测试
### 节点139上运行:
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="192.168.50.139" --master_port=1234 run_mlm.py --model_name_or_path microsoft/deberta-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir output/mydeberta
### 节点169上运行:
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.50.139" --master_port=1234 run_mlm.py --model_name_or_path microsoft/deberta-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir output/mydeberta


# 参数
```buildoutcfg
usage: run_mlm.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH]
                  [--model_type MODEL_TYPE] [--config_name CONFIG_NAME]
                  [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR]
                  [--no_use_fast_tokenizer]
                  [--use_fast_tokenizer [USE_FAST_TOKENIZER]]
                  [--dataset_name DATASET_NAME]
                  [--dataset_config_name DATASET_CONFIG_NAME]
                  [--train_file TRAIN_FILE]
                  [--validation_file VALIDATION_FILE]
                  [--overwrite_cache [OVERWRITE_CACHE]]
                  [--max_seq_length MAX_SEQ_LENGTH]
                  [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]
                  [--mlm_probability MLM_PROBABILITY]
                  [--line_by_line [LINE_BY_LINE]]
                  [--pad_to_max_length [PAD_TO_MAX_LENGTH]]
                  [--output_dir OUTPUT_DIR]
                  [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]]
                  [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]]
                  [--do_predict [DO_PREDICT]]
                  [--evaluation_strategy {no,steps,epoch}]
                  [--prediction_loss_only [PREDICTION_LOSS_ONLY]]
                  [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                  [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                  [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                  [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                  [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                  [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS]
                  [--learning_rate LEARNING_RATE]
                  [--weight_decay WEIGHT_DECAY] [--adam_beta1 ADAM_BETA1]
                  [--adam_beta2 ADAM_BETA2] [--adam_epsilon ADAM_EPSILON]
                  [--max_grad_norm MAX_GRAD_NORM]
                  [--num_train_epochs NUM_TRAIN_EPOCHS]
                  [--max_steps MAX_STEPS]
                  [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                  [--warmup_steps WARMUP_STEPS] [--logging_dir LOGGING_DIR]
                  [--logging_first_step [LOGGING_FIRST_STEP]]
                  [--logging_steps LOGGING_STEPS] [--save_steps SAVE_STEPS]
                  [--save_total_limit SAVE_TOTAL_LIMIT] [--no_cuda [NO_CUDA]]
                  [--seed SEED] [--fp16 [FP16]]
                  [--fp16_opt_level FP16_OPT_LEVEL]
                  [--fp16_backend {auto,amp,apex}] [--local_rank LOCAL_RANK]
                  [--tpu_num_cores TPU_NUM_CORES]
                  [--tpu_metrics_debug [TPU_METRICS_DEBUG]] [--debug [DEBUG]]
                  [--dataloader_drop_last [DATALOADER_DROP_LAST]]
                  [--eval_steps EVAL_STEPS]
                  [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                  [--past_index PAST_INDEX] [--run_name RUN_NAME]
                  [--disable_tqdm DISABLE_TQDM] [--no_remove_unused_columns]
                  [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]]
                  [--label_names LABEL_NAMES [LABEL_NAMES ...]]
                  [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
                  [--metric_for_best_model METRIC_FOR_BEST_MODEL]
                  [--greater_is_better GREATER_IS_BETTER]
                  [--ignore_data_skip [IGNORE_DATA_SKIP]]
                  [--sharded_ddp [SHARDED_DDP]] [--deepspeed DEEPSPEED]
                  [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
                  [--adafactor [ADAFACTOR]]
                  [--group_by_length [GROUP_BY_LENGTH]]
                  [--report_to REPORT_TO [REPORT_TO ...]]
                  [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
                  [--no_dataloader_pin_memory]
                  [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]]
```