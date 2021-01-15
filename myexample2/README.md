## Language model training语言模型训练

对GPT，GPT-2，ALBERT，BERT，DistilBERT，RoBERTa，XLNet...,GPT的文本数据集上用于语言模型的库模型进行微调(或从头开始进行训练)，
GPT-2训练或微调使用因果语言模型(CLM)损失，
而ALBERT，BERT，DistilBERT和RoBERTa则使用Masked语言模型(MLM)损失进行训练或微调。 
XLNet使用排列语言模型(PLM)，您可以在我们的[model summary](https://huggingface.co/transformers/model_summary.html)中找到有关这些目标之间差异的更多信息。 

这些脚本利用了🤗数据集库和Trainer API。 如果您需要对数据集进行额外处理，则可以轻松地根据需要自定义它们。 

**注意:**旧脚本`run_language_modeling.py`仍然可用
[here](https://github.com/huggingface/transformers/blob/master/examples/contrib/legacy/run_language_modeling.py).

以下示例将在我们的[hub](https://huggingface.co/datasets)上托管的数据集上运行，
或与您自己的文本文件一起进行训练和验证。 我们在下面给出两个样本。

### GPT-2/GPT and causal language modeling

下面的样本在WikiText-2上微调GPT-2。 我们正在使用原始的WikiText-2(在tokenization之前没有替换任何tokens)。 这里的损失是因果语言模型的损失。


```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

在单个K80 GPU上训练大约需要半小时，而评估运行大约需要一分钟。 在数据集上进行微调后，它的perplexity约为20。
```buildoutcfg
K80 GPU
    4992 NVIDIA CUDA cores with a dual-GPU design
    Up to 2.91 teraflops double-precision performance with NVIDIA GPU Boost
    Up to 8.73 teraflops single-precision performance with NVIDIA GPU Boost
    24 GB of GDDR5 memory
    480 GB/s aggregate memory bandwidth
    ECC protection for increased reliability
    Server-optimised to deliver the best throughput in the data center
```


要在您自己的训练和验证文件上运行，请使用以下命令：

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```


### RoBERTa/BERT/DistilBERT and masked language modeling

下面的样本在WikiText-2上微调RoBERTa。 在这里，我们也使用原始的WikiText-2。 
由于BERT/RoBERTa具有双向机制，因此损失有所不同。 
因此，我们使用的是与预训练相同的损失：mask语言模型。

根据RoBERTa的论文，我们使用动态masking而不是静态masking。 
因此，该模型的收敛速度可能会稍慢(过拟合会花费更多时间)。

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```
要在您自己的训练和验证文件上运行，请使用以下命令：

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

如果您的数据集每行仅包含一个样本，则可以使用--line_by_line标志
(否则脚本将所有文本拼接在一起，然后将它们分成相同长度的块)。 

**Note:** 在TPU上，应将标志--pad_to_max_length与标志--line_by_line结合使用，以确保所有批次的长度都相同。 

### Whole word masking

BERT作者于2019年5月使用全词mask发布了BERT的新版本。
他们没有mask随机选择的token(可能是单词的一部分)，
而是mask了随机选择的单词(mask了与该单词相对应的所有token)。 
[this paper](https://arxiv.org/abs/1906.08101)针对中文改进了此技术。 


要使用全词masking对模型进行微调，请使用以下脚本：
```bash
python run_mlm_wwm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm-wwm
```

对于中文模型，我们需要生成一个子词的后半部分的关联文件(需要ltp库)，因为它是在字符级别tokenized的。 

**Q :** 为什么要参考文件？ 

**A :** 假设我们有一个中文句子，例如：`我喜欢你` ，原始中文BERT将token为 
`['我','喜','欢','你']` (character level). 但是 `喜欢` 是一个全词. 对于全词mask方式，
我们需要一个类似的结果  `['我','喜','##欢','你']`, 因此，我们需要一个参考文件来告诉模型应在BERT原始token的哪个位置添加“##”。 

**Q :** Why LTP ?

**A :** 
因为最知名的中文WWM BERT是HIT的[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)。它在像CLUE(Chinese GLUE)。 他们使用LTP，所以如果我们要微调他们的模型，我们需要LTP。 
http://ltp.ai/, 下载ltp的模型文件, 下载路径： https://github.com/HIT-SCIR/ltp/blob/master/MODELS.md，可以不用下载，直接用名称指定，如下示例

现在LTP仅在`transformers == 3.2.0`上才能很好地工作。 因此，我们不会将其添加到requirements.txt。
您需要使用此版本的Transformers创建一个单独的环境，
以运行将创建子词的后半部分的关联文件的`run_chinese_ref.py`脚本。 
该脚本在`examples/contrib`中。 在适当的环境中后，请运行以下命令： 

```bash

python examples/contrib/run_chinese_ref.py \
    --file_name=data/demo.txt \
    --ltp=small \
    --bert=bert-base-chinese \
    --save_path=data/ref.txt

输出类似:
第2个样本是: 吸收效果：等价位小 性价比还行
第2个样本的ltp分词后结果: ['还行', '性价', '吸收', '等价位', '效果']
第2个样本的bert toknizer后结果: [101, 1429, 3119, 3126, 3362, 8038, 5023, 817, 855, 2207, 2595, 817, 3683, 6820, 6121, 102]
第2个样本的bert toknizer被ltp的全词处理后的结果: ['[CLS]', '吸', '##收', '效', '##果', '：', '等', '##价', '##位', '小', '性', '##价', '比', '还', '##行', '[SEP]']
第2个样本的bert的token对应的子词的后半部分的位置的最终的ref_id: [2, 4, 7, 8, 11, 14]
```


然后，您可以像这样运行脚本，会自动根据字典大小重新调整嵌入层的大小： 

```bash
python run_mlm_wwm.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --train_file data/demo.txt \
    --validation_file path_to_validation_file \
    --train_ref_file data/ref.txt \
    --remove_unused_columns
    --validation_ref_file path_to_validation_chinese_ref_file \
    --do_train \
    --do_eval \
    --output_dir output

方法1: 继续训练，注意不要让src/transfromers/trainer.py 的Trainer的_remove_unused_columns函数移除chinese_ref列, 需要使用TrainingArguments中的设为False，不要移除，remove_unused_columns，
这样才能对匹配到DataCollatorForWholeWordMask, 的判断语句if "chinese_ref" in e:
还需更改接下来的一行 len_seq = e["input_ids"].size(0) 为 len_seq = len(e["input_ids"])
python run_mlm_wwm.py \
--model_name_or_path
rbt3
--tokenizer_name
bert_model
--train_file
data/demo.txt
--train_ref_file
data/ref.txt
--do_train
--output_dir
output

方法2： scratch 重新训练
python run_mlm_wwm.py \
--model_type
roberta
--config_name
rbt3/roberta
--tokenizer_name
rbt3/roberta
--train_file
data/demo.txt
--train_ref_file
data/ref.txt
--do_train
--output_dir
output

```

**Note:** 在TPU上，您应该标记`--pad_to_max_length`以确保所有批次的长度都相同。 

### XLNet and permutation language modeling

XLNet使用不同的训练目标，即排列语言模型。 它是一种自回归方法，通过最大化输入序列分解阶数的所有排列上的预期似然性来学习双向上下文。 


我们使用--plm_probability标志来定义排列语言模型的mask token跨度的长度与周围上下文长度的比率。 

--max_span_length`标志还可用于限制排列语言模型的mask token的跨度长度。 

这是在wikitext-2上微调XLNet的方法：

```bash
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

要在您自己的训练和验证文件上对其进行微调，请运行：

```bash
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

如果您的数据集每行仅包含一个样本，则可以使用--line_by_line标志(否则脚本将所有文本拼接在一起，然后将它们分成相同长度的块)。 

**Note:** 在TPU上，应将标志--pad_to_max_length与标志--line_by_line结合使用，以确保所有批次的长度都相同。 
