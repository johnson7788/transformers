# Text classification examples

## PyTorch version

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py).

在GLUE基准上微调用于序列分类的库模型：[General Language Understanding Evaluation](https://gluebenchmark.com/)。 
该脚本可以微调[hub](https://huggingface.co/models) 上的任何模型，也可以用于csv或JSON文件中的您自己的数据(该脚本可能需要进行一些调整。)

GLUE由9个不同的任务组成。 这是在其中之一上运行脚本的方法：

```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

其中任务名称可以是cola，sst2，mrpc，stsb，qqp，mnli，qnli，rte，wnli中的一个。

我们使用先前的命令在基准测试的开发集上获得了以下结果(MRPC和WNLI除外，它们很小，使用了3个epoch)。
训练是seeded的，因此您应该在PyTorch 1.6.0上获得相同的结果(并在不同版本上获得接近的结果)，并给出训练时间以供参考(使用了单个Titan RTX)：


| Task  | Metric                       | Result      | Training time |
|-------|------------------------------|-------------|---------------|
| CoLA  | Matthew's corr               | 56.53       | 3:17          |
| SST-2 | Accuracy                     | 92.32       | 26:06         |
| MRPC  | F1/Accuracy                  | 88.85/84.07 | 2:21          |
| STS-B | Person/Spearman corr.        | 88.64/88.48 | 2:13          |
| QQP   | Accuracy/F1                  | 90.71/87.49 | 2:22:26       |
| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       |
| QNLI  | Accuracy                     | 90.66       | 40:57         |
| RTE   | Accuracy                     | 65.70       | 57            |
| WNLI  | Accuracy                     | 56.34       | 24            |


其中一些结果与网站上GLUE基准测试集上报告的结果有显着差异。 
有关QQP和WNLI，请参阅网站上的[FAQ #12](https://gluebenchmark.com/faq) 。


### 混合精度训练

If you have a GPU with mixed precision capabilities (architecture Pascal or more recent), you can use mixed precision
training with PyTorch 1.6.0 or latest, or by installing the [Apex](https://github.com/NVIDIA/apex) library for previous
versions. Just add the flag `--fp16` to your command launching one of the scripts mentioned above!

Using mixed precision training usually results in 2x-speedup for training with the same final results:

| Task  | Metric                       | Result      | Training time | Result (FP16) | Training time (FP16) |
|-------|------------------------------|-------------|---------------|---------------|----------------------|
| CoLA  | Matthew's corr               | 56.53       | 3:17          | 56.78         | 1:41                 |
| SST-2 | Accuracy                     | 92.32       | 26:06         | 91.74         | 13:11                |
| MRPC  | F1/Accuracy                  | 88.85/84.07 | 2:21          | 88.12/83.58   | 1:10                 |
| STS-B | Person/Spearman corr.        | 88.64/88.48 | 2:13          | 88.71/88.55   | 1:08                 |
| QQP   | Accuracy/F1                  | 90.71/87.49 | 2:22:26       | 90.67/87.43   | 1:11:54              |
| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       | 84.04/84.06   | 1:17:06              |
| QNLI  | Accuracy                     | 90.66       | 40:57         | 90.96         | 20:16                |
| RTE   | Accuracy                     | 65.70       | 57            | 65.34         | 29                   |
| WNLI  | Accuracy                     | 56.34       | 24            | 56.34         | 12                   |


# Run TensorFlow 2.0 version
run_tf_glue.py 是tensorflow版本的GLUE分类
脚本是  [`run_tf_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_glue.py).

对GLUE基准的MRPC任务上的序列分类库TensorFlow 2.0 Bert模型进行微调:  [General Language Understanding Evaluation](https://gluebenchmark.com/).

该脚本具有用于在Tensor Core（NVIDIA Volta / Turing GPU）和将来的硬件上运行模型的混合精度选项（Automatic Mixed Precision / AMP），以及XLA的选项，该选项使用XLA编译器来减少模型运行时间。
在脚本中使用“ USE_XLA”或“ USE_AMP”变量来切换选项。
这些选项和以下基准由@tlkh提供。

脚本的快速基准测试（无其他修改）：

| GPU    | Mode | Time (2nd epoch) | Val Acc (3 runs) |
| --------- | -------- | ----------------------- | ----------------------|
| Titan V | FP32 | 41s | 0.8438/0.8281/0.8333 |
| Titan V | AMP | 26s | 0.8281/0.8568/0.8411 |
| V100    | FP32 | 35s | 0.8646/0.8359/0.8464 |
| V100    | AMP | 22s | 0.8646/0.8385/0.8411 |
| 1080 Ti | FP32 | 55s | - |

对于相同的硬件和超参数（使用相同的批次大小），混合精度（AMP）大大减少了训练时间。

## Run generic text classification script in TensorFlow

The script [run_tf_text_classification.py](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_text_classification.py) allows users to run a text classification on their own CSV files. For now there are few restrictions, the CSV files must have a header corresponding to the column names and not more than three columns: one column for the id, one column for the text and another column for a second piece of text in case of an entailment classification for example.

To use the script, one as to run the following command line:
```bash
python run_tf_text_classification.py \
  --train_file train.csv \ ### training dataset file location (mandatory if running with --do_train option)
  --dev_file dev.csv \ ### development dataset file location (mandatory if running with --do_eval option)
  --test_file test.csv \ ### test dataset file location (mandatory if running with --do_predict option)
  --label_column_id 0 \ ### which column corresponds to the labels
  --model_name_or_path bert-base-multilingual-uncased \
  --output_dir model \
  --num_train_epochs 4 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --do_train \
  --do_eval \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --save_steps 10 \
  --overwrite_output_dir \
  --max_seq_length 128
```
>>>>>>> upstream/master

<<<<<<< HEAD
# Run PyTorch version
脚本是Pytorch版本 [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py).

在GLUE基准上微调用于序列分类的库模型： [General Language Understanding
Evaluation](https://gluebenchmark.com/).
该脚本可以微调以下模型：BERT，XLM，XLNet和RoBERTa。 

GLUE由9个不同的任务组成。我们在不带大小写的BERT基本模型（checkpoint“ bert-base-uncased”）的基准开发集上获得以下结果。
所有实验都运行单个V100 GPU，总训练批大小在16到64之间。
其中一些任务的数据集很小，训练可能导致不同运行之间的结果差异很大。我们针对每个指标报告5次运行（中位数不同）的中位数。

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthew's corr               | 49.23       |
| SST-2 | Accuracy                     | 91.97       |
| MRPC  | F1/Accuracy                  | 89.47/85.29 |
| STS-B | Person/Spearman corr.        | 83.95/83.70 |
| QQP   | Accuracy/F1                  | 88.40/84.31 |
| MNLI  | Matched acc./Mismatched acc. | 80.61/81.08 |
| QNLI  | Accuracy                     | 87.46       |
| RTE   | Accuracy                     | 61.73       |
| WNLI  | Accuracy                     | 45.07       |

<<<<<<< HEAD
其中一些结果与网站上GLUE基准测试集上报告的结果有很大不同。
有关QQP和WNLI，请参阅网站上的  [FAQ #12](https://gluebenchmark.com/faq) on the webite.

在运行这些GLUE任务中的任何一项之前， you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running the following lines at the root of the repo
```
python utils/download_glue_data.py --data_dir /path/to/glue --tasks all
```

替换GLUE_DIR路径
=======
Some of these results are significantly different from the ones reported on the test set
of GLUE benchmark on the website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the
website.
>>>>>>> upstream/master

```bash
export TASK_NAME=MRPC

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/
```
其中任务名称可以是CoLA，SST-2，MRPC，STS-B，QQP，MNLI，QNLI，RTE，WNLI之一。


开发集的结果将出现在指定output_dir中的文本文件eval_results.txt中。
对于MNLI，由于有两个单独的开发集（匹配和不匹配），所以除了/tmp/MNLI/之外，还有一个单独的输出文件夹，称为`/tmp/MNLI-MM/。

除了MRPC，MNLI，CoLA和SST-2之外，还没有对任何GLUE任务进行过半精度训练的测试。
下一节提供有关如何使用MRPC进行半精度训练的详细信息。 
话虽如此，对剩余的GLUE任务进行半精度训练也不应该有任何问题，因为每个任务的数据处理器都继承自基类DataProcessor。


## Running on TPUs in PyTorch

<<<<<<< HEAD
**Update**: read the more up-to-date [Running on TPUs](../README.md#running-on-tpus) in the main README.md instead.

即使在运行PyTorch时，也可以使用pytorch / xla加快Google TPU上的工作量。 有关如何设置TPU环境的信息， using `pytorch/xla`. 请参阅
[pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

以下是在TPU上运行`* _tpu.py`优化脚本的一些示例。数据准备的所有步骤均与常规GPU + Huggingface设置相同。

为了在MNLI数据集上运行GLUE任务，您可以运行以下内容：
=======
Even when running PyTorch, you can accelerate your workloads on Google's TPUs, using `pytorch/xla`. For information on
how to setup your TPU environment refer to the
[pytorch/xla README](https://github.com/pytorch/xla/blob/master/README.md).

For running your GLUE task on MNLI dataset you can run something like the following form the root of the transformers
repo:
>>>>>>> upstream/master

```
python examples/xla_spawn.py \
  --num_cores=8 \
  transformers/examples/text-classification/run_glue.py \
  --do_train \
  --do_eval \
  --task_name=mrpc \
  --num_train_epochs=3 \
  --max_seq_length=128 \
  --learning_rate=5e-5 \
  --output_dir=/tmp/mrpc \
  --overwrite_output_dir \
  --logging_steps=5 \
  --save_steps=5 \
  --tpu_metrics_debug \
  --model_name_or_path=bert-base-cased \
  --per_device_train_batch_size=64 \
  --per_device_eval_batch_size=64
```

<<<<<<< HEAD
### MRPC

#### Fine-tuning example

以下示例对Microsoft Research Paraphrase语料库（MRPC）语料库上的BERT进行了微调，
并且在单个K-80上运行不到10分钟，而在安装了apex的单个tesla V100 16GB上运行了27秒。

在运行这些GLUE任务中的任何一项之前，您应该下载， 通过运行
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```bash
export GLUE_DIR=/path/to/glue

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
```

我们的测试基于 [the original implementation hyper-
parameters](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks) 
评估结果介于84％和88％之间。
=======
>>>>>>> upstream/master

#### Using Apex and mixed-precision

使用Apex和16位精度，在MRPC上的微调仅需27秒。 First install
[apex](https://github.com/NVIDIA/apex), then run the following example:

```bash

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/ \
  --fp16
```

#### Distributed training
这是在8个V100 GPU上使用分布式训练的示例。使用的模型是BERTwhole-word-masking，在MRPC上达到F1> 92


```bash

python -m torch.distributed.launch \
    --nproc_per_node 8 run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mrpc \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /tmp/mrpc_output/
```

使用这些超参数进行训练可以为我们带来以下结果：


```bash
acc = 0.8823529411764706
acc_and_f1 = 0.901702786377709
eval_loss = 0.3418912578906332
f1 = 0.9210526315789473
global_step = 174
loss = 0.07231863956341798
```

### MNLI

下面的示例使用BERT大型，BERT-large, uncased, whole-word-masking model，并在MNLI任务上对其进行微调。

```bash
export GLUE_DIR=/path/to/glue

python -m torch.distributed.launch \
    --nproc_per_node 8 run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mnli \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir output_dir \
```

The results  are the following:

```bash
***** Eval results *****
  acc = 0.8679706601466992
  eval_loss = 0.4911287787382479
  global_step = 18408
  loss = 0.04755385363816904

***** Eval results *****
  acc = 0.8747965825874695
  eval_loss = 0.45516540421714036
  global_step = 18408
  loss = 0.04755385363816904
```

# Run PyTorch version using PyTorch-Lightning

<<<<<<< HEAD
从`glue`目录运行`bash run_pl.sh`。 这也将安装`pytorch-lightning`和`examples / requirements.txt`中的要求。 这是一个shell管道，将自动下载，预处理数据并运行指定的模型。 日志保存在“ lightning_logs”目录中。
=======
Run `bash run_pl.sh` from the `glue` directory. This will also install `pytorch-lightning` and the requirements in
`examples/requirements.txt`. It is a shell pipeline that will automatically download, preprocess the data and run the
specified models. Logs are saved in `lightning_logs` directory.
>>>>>>> upstream/master

传递--gpus标志来更改GPU的数量。 默认使用1.最后，预期结果是：

```
TEST RESULTS {'val_loss': tensor(0.0707), 'precision': 0.852427800698191, 'recall': 0.869537067011978, 'f1': 0.8608974358974358}
```

=======
>>>>>>> DataCollatorForWholeWordMask_Bug

## XNLI

Based on the script [`run_xnli.py`](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_xnli.py).

[XNLI](https://www.nyu.edu/projects/bowman/xnli/) is a crowd-sourced dataset based on [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/). 
它是跨语言文本表示形式的评估基准。成对的文本用15种不同语言（包括high-resource language （例如英语）和low-resource languages （例如Swahili语））的文本包含注释标记。

#### Fine-tuning on XNLI

此示例代码在XNLI数据集上微调了mBERT（多语言BERT）。 它在单个tesla V100 16GB上运行106分钟。 
可以通过以下链接下载XNLI的数据，并且应将其同时保存（并解压缩）在`$XNLI_DIR`目录中。


* [XNLI 1.0](https://cims.nyu.edu/~sbowman/xnli/XNLI-1.0.zip)
* [XNLI-MT 1.0](https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip)

```bash
export XNLI_DIR=/path/to/XNLI

python run_xnli.py \
  --model_name_or_path bert-base-multilingual-cased \
  --language de \
  --train_language en \
  --do_train \
  --do_eval \
  --data_dir $XNLI_DIR \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir /tmp/debug_xnli/ \
  --save_steps -1
```

使用先前定义的超参数进行训练会在**test**集合上产生以下结果：

```bash
acc = 0.7093812375249501
```


# 句子连贯性分类
python sequence_classfication.py --model_name_or_path albert-base-v2 --task_name smooth --task_script data/smooth.py \
--output_dir smooth --do_train --do_eval --max_seq_length 64 --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 3
