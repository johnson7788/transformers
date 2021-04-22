# seq2seq
## 序列序列训练和评估 

这个目录包含了对总结和翻译任务的transformer进行微调和评估的例子。
Please tag @patil-suraj with any issues/unexpected behaviors, or send a PR!
For deprecated `bertabs` instructions, see [`bertabs/README.md`](https://github.com/huggingface/transformers/blob/master/examples/research_projects/bertabs/README.md).
For the old `finetune_trainer.py` and related utils, see [`examples/legacy/seq2seq`](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq).

### 支持的架构 

- `BartForConditionalGeneration`
- `FSMTForConditionalGeneration` (translation only)
- `MBartForConditionalGeneration`
- `MarianMTModel`
- `PegasusForConditionalGeneration`
- `T5ForConditionalGeneration`


`run_summarization.py`和`run_translation.py`是轻量级的例子，说明如何从[🤗 Datasets](https://github.com/huggingface/datasets)库中下载和预处理数据集，或者使用你自己的文件(jsonlines或csv)，然后对其进行上述架构之一进行微调。



For custom datasets in `jsonlines` format please see: https://huggingface.co/docs/datasets/loading_datasets.html#json-files
and you also will find examples of these below.

关于 "jsonlines "格式的自定义数据集，请参见：https://huggingface.co/docs/datasets/loading_datasets.html#json-files。
而你也会在下面找到这些例子。

### 摘要汇总summarization任务

以下是摘要任务的样本：

```bash
python examples/seq2seq/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

只有 T5 models `t5-small`, `t5-base`, `t5-large`, `t5-3b` and `t5-11b` 需要附加参数: `--source_prefix "summarize: "`.

我们在这个例子中使用了CNN/DailyMail数据集，因为 "t5-small "在上面进行了训练，即使用很小的样本进行预训练，也能得到不错的分数。

Extreme Summarization（XSum）数据集是另一个常用的数据集，用于总结任务。要使用它，将`--dataset_name cnn_dailymail --dataset_config "3.0.0"`替换为`--dataset_name xsum`。

使用自己的训练文件，使用以下参数
`--train_file`, `--validation_file`, `--text_column` and `--summary_column` to match your setup:

```bash
python examples/seq2seq/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

摘要任务支持自定义CSV和JSonlines格式。

#### 自定义CSV文件

如果是csv文件，训练和验证文件应该有一列输入文本和一列摘要。
如果csv文件只有两列，如下面的例子。


```csv
text,summary
"I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder","I'm sitting in a room where I'm waiting for something to happen"
"I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.","I'm a gardener and I'm a big fan of flowers."
"Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share","It's that time of year again."
```

第一列假定为 "文本"，第二列为摘要。
如果csv文件有多列，你可以指定要使用的列名。

```bash
    --text_column text_column_name \
    --summary_column summary_column_name \
```

例如，如果列是： 

```csv
id,date,text,summary
```

如果你想只选择`text`和`summary`，那么你将传递这些附加参数。

```bash
    --text_column text \
    --summary_column summary \
```

#### 自定义jsonlines文件 

第二种支持的格式是jsonlines。下面是一个jsonlines自定义数据文件的例子。

```json
{"text": "I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder", "summary": "I'm sitting in a room where I'm waiting for something to happen"}
{"text": "I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.", "summary": "I'm a gardener and I'm a big fan of flowers."}
{"text": "Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share", "summary": "It's that time of year again."}
```

与CSV文件一样，默认情况下，第一个值将作为文本记录，第二个值作为摘要记录。因此，你可以使用任何键名作为条目，在本例中使用了 "text "和 "summary"。

和CSV文件一样，你可以通过明确地指定相应的键名，来指定从文件中选择哪些值。在我们的例子中，这又将是。

```bash
    --text_column text \
    --summary_column summary \
```



### 翻译 Translation

下面是一个用MarianMT模型进行翻译微调的例子。

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-ro \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

MBART和一些T5模型需要特殊处理。

T5 models `t5-small`, `t5-base`, `t5-large`, `t5-3b` and `t5-11b` 必须使用额外的参数: `--source_prefix "translate {source_lang} to {target_lang}"`. For example:

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

如果你得到一个糟糕的BLEU分数，请确保你没有忘记使用`--source_prefix`参数。

对于上述一组T5模型，需要记住的是，如果切换到不同的语言对，一定要调整3个特定语言命令行参数中的源和目标值：`--source_lang`、`--target_lang`和`--source_prefix`。

# Mbart
MBart模型要求"--source_lang "和"--target_lang "的值采用不同的格式，例如，它希望用 "en_XX "代替 "en"，对于 "ro"，它希望用 "ro_RO"。完整的MBart语言代码规范可以在[这里](https://huggingface.co/facebook/mbart-large-cc25)找到。例如：

export langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path facebook/mbart-large-en-ro  \
    --do_train \
    --do_eval \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en_XX \
    --target_lang ro_RO \
    --output_dir output/en-ro-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate

需要下载数据集225M+23.5M+38.7M, max_train_samples, max_val_samples, max_test_samples 在调试时使用
python run_translation.py
--model_name_or_path
facebook/mbart-large-en-ro
--do_train
--do_eval
--dataset_name
wmt16/wmt16.py
--dataset_config_name
ro-en
--source_lang
en_XX
--target_lang
ro_RO
--output_dir
output/en-ro-translation
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--overwrite_output_dir
--predict_with_generate
--max_train_samples=2000
--max_val_samples=500
--max_test_samples=200
```
#使用facebook/mbart-large-cc25微调模型，fp16，12G显存不够用
python run_translation.py  --model_name_or_path facebook/mbart-large-cc25  \
    --do_train \
    --do_eval \
    --fp16 True \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en_XX \
    --target_lang ro_RO \
    --output_dir tst-translation \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples=2000 \
    --max_val_samples=500 \
    --max_test_samples=200
    
# mbart50     facebook/mbart-large-50 还是显存溢出
python run_translation.py  --model_name_or_path facebook/mbart-large-50  \
    --do_train \
    --do_eval \
    --fp16 True \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en_XX \
    --target_lang ro_RO \
    --output_dir tst-translation \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples=2000 \
    --max_val_samples=500 \
    --max_test_samples=200

# m2m100模型 使用模型facebook/m2m100_418M, 注意source_lang和target_lang发生了改变，是不一样的格式， 英语到罗马尼亚语， 12GB显存不会溢出，很不错
python run_translation.py  --model_name_or_path facebook/m2m100_418M  \
    --do_train \
    --do_eval \
    --fp16 True \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en \
    --target_lang ro \
    --output_dir tst-translation \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_train_samples=2000 \
    --max_val_samples=500 \
    --max_test_samples=200

# 训练中文到英文, 使用wmt19数据集，使用下载好的本地模型facebook/mbart-large-cc25, 需要中英文平行语料下载，网络不太ok, 需要下载的数据集很多
使用了m2m100的多语言模型，需要搭配使用--forced_bos_token参数，表明第一个token是生成的目标语言的种类
mbart是自动设置了decoder_start_token_id
```
'newscommentary_v14' = {list: 1} ['http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-zh.tsv.gz']
'wikititles_v1' = {list: 1} ['http://data.statmt.org/wikititles/v1/wikititles-v1.zh-en.tsv.gz']
'uncorpus_v1' = {list: 1} ['https://storage.googleapis.com/tfdataset-data/downloadataset/uncorpus/UNv1.0.en-zh.tar.gz']
'casia2015' = {list: 1} ['ftp://cwmt-wmt:cwmt-wmt@datasets.nju.edu.cn/parallel/casia2015.zip']
'casict2011' = {list: 1} ['ftp://cwmt-wmt:cwmt-wmt@datasets.nju.edu.cn/parallel/casict2011.zip']
'casict2015' = {list: 1} ['ftp://cwmt-wmt:cwmt-wmt@datasets.nju.edu.cn/parallel/casict2015.zip']
'datum2015' = {list: 1} ['ftp://cwmt-wmt:cwmt-wmt@datasets.nju.edu.cn/parallel/datum2015.zip']
'datum2017' = {list: 1} ['ftp://cwmt-wmt:cwmt-wmt@datasets.nju.edu.cn/parallel/datum2017.zip']
'neu2017' = {list: 1} ['ftp://cwmt-wmt:cwmt-wmt@datasets.nju.edu.cn/parallel/neu2017.zip']
'newstest2018' = {list: 1} ['http://data.statmt.org/wmt19/translation-task/dev.tgz']
```
```
python examples/seq2seq/run_translation.py --model_name_or_path mbart-local --do_train --do_eval 
--dataset_name wmt19  --dataset_config_name zh-en --source_lang zh_CN --target_lang en_XX 
--output_dir output/zh-en-translation --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate
 ```

在调整参数`--train_file`、`--validation_file`的值以符合你的设置后，你将如何在自己的文件上使用翻译微调。

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --train_file path_to_jsonlines_file \
    --validation_file path_to_jsonlines_file \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

翻译的任务只支持自定义的JSONLINES文件，每一行都是一个键为 "翻译 "的字典，其值为另一个键为语言对的字典。例如：

```json
{ "translation": { "en": "Others have dismissed him as a joke.", "ro": "Alții l-au numit o glumă." } }
{ "translation": { "en": "And some are holding out for an implosion.", "ro": "Iar alții așteaptă implozia." } }
```
这里的语言是罗马尼亚语（`ro`）和英语（`en`）。

如果你想使用一个预处理数据集，导致高BLEU分数，但对于`en-de`语言对，你可以使用`--dataset_name stas/wmt14-en-de-预处理`，如下。

```bash
python examples/seq2seq/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name stas/wmt14-en-de-pre-processed \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
 ```


# 命令帮助  
usage: run_translation.py [-h] --model_name_or_path MODEL_NAME_OR_PATH
                          [--config_name CONFIG_NAME]
                          [--tokenizer_name TOKENIZER_NAME]
                          [--cache_dir CACHE_DIR] [--no_use_fast_tokenizer]
                          [--use_fast_tokenizer [USE_FAST_TOKENIZER]]
                          [--model_revision MODEL_REVISION]
                          [--use_auth_token [USE_AUTH_TOKEN]]
                          [--source_lang SOURCE_LANG]
                          [--target_lang TARGET_LANG]
                          [--dataset_name DATASET_NAME]
                          [--dataset_config_name DATASET_CONFIG_NAME]
                          [--train_file TRAIN_FILE]
                          [--validation_file VALIDATION_FILE]
                          [--test_file TEST_FILE]
                          [--overwrite_cache [OVERWRITE_CACHE]]
                          [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]
                          [--max_source_length MAX_SOURCE_LENGTH]
                          [--max_target_length MAX_TARGET_LENGTH]
                          [--val_max_target_length VAL_MAX_TARGET_LENGTH]
                          [--pad_to_max_length [PAD_TO_MAX_LENGTH]]
                          [--max_train_samples MAX_TRAIN_SAMPLES]
                          [--max_val_samples MAX_VAL_SAMPLES]
                          [--max_test_samples MAX_TEST_SAMPLES]
                          [--num_beams NUM_BEAMS]
                          [--no_ignore_pad_token_for_loss]
                          [--ignore_pad_token_for_loss [IGNORE_PAD_TOKEN_FOR_LOSS]]
                          [--source_prefix SOURCE_PREFIX]
                          [--forced_bos_token FORCED_BOS_TOKEN] --output_dir
                          OUTPUT_DIR
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
                          [--weight_decay WEIGHT_DECAY]
                          [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                          [--adam_epsilon ADAM_EPSILON]
                          [--max_grad_norm MAX_GRAD_NORM]
                          [--num_train_epochs NUM_TRAIN_EPOCHS]
                          [--max_steps MAX_STEPS]
                          [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                          [--warmup_ratio WARMUP_RATIO]
                          [--warmup_steps WARMUP_STEPS]
                          [--logging_dir LOGGING_DIR]
                          [--logging_strategy {no,steps,epoch}]
                          [--logging_first_step [LOGGING_FIRST_STEP]]
                          [--logging_steps LOGGING_STEPS]
                          [--save_strategy {no,steps,epoch}]
                          [--save_steps SAVE_STEPS]
                          [--save_total_limit SAVE_TOTAL_LIMIT]
                          [--no_cuda [NO_CUDA]] [--seed SEED] [--fp16 [FP16]]
                          [--fp16_opt_level FP16_OPT_LEVEL]
                          [--fp16_backend {auto,amp,apex}]
                          [--fp16_full_eval [FP16_FULL_EVAL]]
                          [--local_rank LOCAL_RANK]
                          [--tpu_num_cores TPU_NUM_CORES]
                          [--tpu_metrics_debug [TPU_METRICS_DEBUG]]
                          [--debug [DEBUG]]
                          [--dataloader_drop_last [DATALOADER_DROP_LAST]]
                          [--eval_steps EVAL_STEPS]
                          [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                          [--past_index PAST_INDEX] [--run_name RUN_NAME]
                          [--disable_tqdm DISABLE_TQDM]
                          [--no_remove_unused_columns]
                          [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]]
                          [--label_names LABEL_NAMES [LABEL_NAMES ...]]
                          [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
                          [--metric_for_best_model METRIC_FOR_BEST_MODEL]
                          [--greater_is_better GREATER_IS_BETTER]
                          [--ignore_data_skip [IGNORE_DATA_SKIP]]
                          [--sharded_ddp SHARDED_DDP] [--deepspeed DEEPSPEED]
                          [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
                          [--adafactor [ADAFACTOR]]
                          [--group_by_length [GROUP_BY_LENGTH]]
                          [--length_column_name LENGTH_COLUMN_NAME]
                          [--report_to REPORT_TO [REPORT_TO ...]]
                          [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
                          [--no_dataloader_pin_memory]
                          [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]]
                          [--skip_memory_metrics [SKIP_MEMORY_METRICS]]
                          [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]]
                          [--mp_parameters MP_PARAMETERS]
                          [--sortish_sampler [SORTISH_SAMPLER]]
                          [--predict_with_generate [PREDICT_WITH_GENERATE]]

optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        来自huggingface.co/models的预训练模型或模型标识符的路径
  --config_name CONFIG_NAME
                        预训练模型的配置名称或路径如果None，那么等同于model_name
  --tokenizer_name TOKENIZER_NAME
                        预训练模型的tokenizer名称或路径如果None，那么等同于model_name
  --cache_dir CACHE_DIR
                        本地路径：在哪里存储从HuggingFace.co下载的预磨模模型
  --no_use_fast_tokenizer
                        是否使用其中一个fast tokenizer（由tokenizer库支持）
  --use_fast_tokenizer [USE_FAST_TOKENIZER]
                        是否使用其中一个fast tokenizer（由tokenizer库支持）
  --model_revision MODEL_REVISION
                        要使用的特定模型版本（可以是分支名称、标签名称或提交ID）。
  --use_auth_token [USE_AUTH_TOKEN]
                        将使用运行`transformers-cli
                        login`时生成的token（在私有模型中使用此脚本时必须使用）。
  --source_lang SOURCE_LANG
                        翻译的源语言ID。
  --target_lang TARGET_LANG
                        翻译目标语言ID。
  --dataset_name DATASET_NAME
                        要使用的数据集的名称（通过datasets库）
  --dataset_config_name DATASET_CONFIG_NAME
                        要使用的数据集的配置名称（通过datasets库）。
  --train_file TRAIN_FILE
                        输入训练数据文件（jsonlines）。
  --validation_file VALIDATION_FILE
                        一个可选的输入评估数据文件，用于评估jsonlines文件上的指标（sacreblue）
  --test_file TEST_FILE
                        一个可选的输入测试数据文件，用于评估jsonlines文件的指标（sacreblue）。
  --overwrite_cache [OVERWRITE_CACHE]
                        覆盖缓存的训练和评估集
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        用于预处理的进程数量。
  --max_source_length MAX_SOURCE_LENGTH
                        token化后的最大输入序列长度。长于这个长度的序列将被截断，短于这个长度的序列将被填充。
  --max_target_length MAX_TARGET_LENGTH
                        token化后的最大输入序列长度。长于这个长度的序列将被截断，短于这个长度的序列将被填充。
  --val_max_target_length VAL_MAX_TARGET_LENGTH
                        token化后的最大输入序列长度。长于这个长度的序列将被截断，短于这个长度的序列将被填充。将默认为`max_
                        target_length`。这个参数也被用来覆盖``model.generate'的``max_lengt
                        h'参数，在``evaluate'和``predict'时使用。
  --pad_to_max_length [PAD_TO_MAX_LENGTH]
                        是否将所有样本Padding到最大句子长度的模型上。如果是False，在批次时将动态地将样本Padding到
                        批中的最大长度。在GPU上更有效，但对TPU非常不利。
  --max_train_samples MAX_TRAIN_SAMPLES
                        为了调试的目的或更快的训练，如果设置了训练实例的数量，就将其截断为这个值。
  --max_val_samples MAX_VAL_SAMPLES
                        为了调试的目的或更快的训练，如果设置了验证实例的数量，则将其截断为这个值。
  --max_test_samples MAX_TEST_SAMPLES
                        为了调试的目的或更快的训练，如果设置了测试实例的数量，就将其截断为这个值。
  --num_beams NUM_BEAMS
                        用于评估的beams的数量。这个参数将被传递给 `model.generate`，在 `evaluate`和
                        `predict`中使用。
  --no_ignore_pad_token_for_loss
                        在损失计算中是否忽略与填充标签相对应的token。
  --ignore_pad_token_for_loss [IGNORE_PAD_TOKEN_FOR_LOSS]
                        在损失计算中是否忽略与填充标签相对应的token。
  --source_prefix SOURCE_PREFIX
                        在每个源文本前添加一个前缀（对T5模型有用）。
  --forced_bos_token FORCED_BOS_TOKEN
                        在:obj:`decoder_start_token_id`之后，强制作为第一个生成的token。对于多语言
                        模型，如:doc:`mBART <.../model_doc/mbart>`，第一个生成的token需要是目
                        标语言的token（通常是目标语言token）。
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and
                        checkpoints will be written.
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory.Use this
                        to continue training if output_dir points to a
                        checkpoint directory.
  --do_train [DO_TRAIN]
                        Whether to run training.
  --do_eval [DO_EVAL]   Whether to run eval on the dev set.
  --do_predict [DO_PREDICT]
                        Whether to run predictions on the test set.
  --evaluation_strategy {no,steps,epoch}
                        The evaluation strategy to use.
  --prediction_loss_only [PREDICTION_LOSS_ONLY]
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
                        The initial learning rate for AdamW.
  --weight_decay WEIGHT_DECAY
                        Weight decay for AdamW if we apply some.
  --adam_beta1 ADAM_BETA1
                        Beta1 for AdamW optimizer
  --adam_beta2 ADAM_BETA2
                        Beta2 for AdamW optimizer
  --adam_epsilon ADAM_EPSILON
                        Epsilon for AdamW optimizer.
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform.
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform.
                        Override num_train_epochs.
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use.
  --warmup_ratio WARMUP_RATIO
                        Linear warmup over warmup_ratio fraction of total
                        steps.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --logging_dir LOGGING_DIR
                        Tensorboard log dir.
  --logging_strategy {no,steps,epoch}
                        The logging strategy to use.
  --logging_first_step [LOGGING_FIRST_STEP]
                        Log the first global_step
  --logging_steps LOGGING_STEPS
                        Log every X updates steps.
  --save_strategy {no,steps,epoch}
                        The checkpoint save strategy to use.
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps.
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints.Deletes the
                        older checkpoints in the output_dir. Default is
                        unlimited checkpoints
  --no_cuda [NO_CUDA]   Do not use CUDA even when it is available
  --seed SEED           Random seed that will be set at the beginning of
                        training.
  --fp16 [FP16]         Whether to use 16-bit (mixed) precision instead of
                        32-bit
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in
                        ['O0', 'O1', 'O2', and 'O3'].See details at
                        https://nvidia.github.io/apex/amp.html
  --fp16_backend {auto,amp,apex}
                        The backend to be used for mixed precision.
  --fp16_full_eval [FP16_FULL_EVAL]
                        Whether to use full 16-bit precision evaluation
                        instead of 32-bit
  --local_rank LOCAL_RANK
                        For distributed training: local_rank
  --tpu_num_cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by
                        launcher script)
  --tpu_metrics_debug [TPU_METRICS_DEBUG]
                        Deprecated, the use of `--debug` is preferred. TPU:
                        Whether to print debug metrics
  --debug [DEBUG]       Whether to print debug metrics on TPU
  --dataloader_drop_last [DATALOADER_DROP_LAST]
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
  --remove_unused_columns [REMOVE_UNUSED_COLUMNS]
                        Remove columns not required by the model when using an
                        nlp.Dataset.
  --label_names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that
                        correspond to the labels.
  --load_best_model_at_end [LOAD_BEST_MODEL_AT_END]
                        Whether or not to load the best model found during
                        training at the end of training.
  --metric_for_best_model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models.
  --greater_is_better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be
                        maximized or not.
  --ignore_data_skip [IGNORE_DATA_SKIP]
                        When resuming training, whether or not to skip the
                        first epochs and batches to get to the same training
                        data.
  --sharded_ddp SHARDED_DDP
                        Whether or not to use sharded DDP training (in
                        distributed training only). The base option should be
                        `simple`, `zero_dp_2` or `zero_dp_3` and you can add
                        CPU-offload to `zero_dp_2` or `zero_dp_3` like this:
                        zero_dp_2 offload` or `zero_dp_3 offload`. You can add
                        auto-wrap to `zero_dp_2` or with the same syntax:
                        zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.
  --deepspeed DEEPSPEED
                        Enable deepspeed and pass the path to deepspeed json
                        config file (e.g. ds_config.json) or an already loaded
                        json file as a dict
  --label_smoothing_factor LABEL_SMOOTHING_FACTOR
                        The label smoothing epsilon to apply (zero means no
                        label smoothing).
  --adafactor [ADAFACTOR]
                        Whether or not to replace AdamW by Adafactor.
  --group_by_length [GROUP_BY_LENGTH]
                        Whether or not to group samples of roughly the same
                        length together when batching.
  --length_column_name LENGTH_COLUMN_NAME
                        Column name with precomputed lengths to use when
                        grouping by length.
  --report_to REPORT_TO [REPORT_TO ...]
                        The list of integrations to report the results and
                        logs to.
  --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS
                        When using distributed training, the value of the flag
                        `find_unused_parameters` passed to
                        `DistributedDataParallel`.
  --no_dataloader_pin_memory
                        Whether or not to pin memory for DataLoader.
  --dataloader_pin_memory [DATALOADER_PIN_MEMORY]
                        Whether or not to pin memory for DataLoader.
  --skip_memory_metrics [SKIP_MEMORY_METRICS]
                        Whether or not to skip adding of memory profiler
                        reports to metrics.
  --use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]
                        Whether or not to use the legacy prediction_loop in
                        the Trainer.
  --mp_parameters MP_PARAMETERS
                        Used by the SageMaker launcher to send mp-specific
                        args. Ignored in Trainer
  --sortish_sampler [SORTISH_SAMPLER]
                        Whether to use SortishSampler or not.
  --predict_with_generate [PREDICT_WITH_GENERATE]
                        Whether to use generate to calculate generative
                        metrics (ROUGE, BLEU).