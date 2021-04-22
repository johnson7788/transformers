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
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate

#训练中文到英文, 使用wmt19数据集，使用下载好的本地模型facebook/mbart-large-cc25, 需要中英文平行语料下载，网络不太ok, 需要下载的数据集很多
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
