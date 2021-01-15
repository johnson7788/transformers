# Plug and Play Language Models: PPLM 简单的控制语言生成方向的模型方法
Authors: [Sumanth Dathathri](https://dathath.github.io/), [Andrea Madotto](https://andreamad8.github.io/), Janice Lan, Jane Hung, Eric Frank, [Piero Molino](https://w4nderlu.st/), [Jason Yosinski](http://yosinski.com/), and [Rosanne Liu](http://www.rosanneliu.com/)
This folder contains the original code used to run the Plug and Play Language Model (PPLM).
Paper link: https://arxiv.org/abs/1912.02164
Blog link: https://eng.uber.com/pplm
Please check out the repo under uber-research for more information: https://github.com/uber-research/PPLM

## 原理
GPT2属于 p(x) 生成的文本是无条件的概率分布,无法控制其生成方向
p(x|a) 是我们要创建的LM语言模型，就是带有属性的文本，偏向某种感情或主题

三个步骤：
1. 使用语言模型transformer进行前向传递，以使用预测p（a | x）的属性模型来计算所需属性的可能性
2. 反向传播基于属性模型中的梯度来更新LM的内部潜在表示，这里是更新transformer的K,V, 以便增加所生成的遍历具有所需属性的可能性
3. 生成词汇表上的新分布,根据新的分布生成新词 

保持修改后的语言模型transformer和原始语言模型GPT-2生成的流畅度相似，损失函数是通过计算2个分布的KL散度

属性模型：
方法1: 词袋法   简单直观，无法表达属性
方法2: PPLM-Discrim， 分为单属性和多属性， 单属性，例如积极或消极，多属性，例如积极的政治方向 


## 安装方法

```bash
git clone https://github.com/huggingface/transformers && cd transformers
pip install [--editable] .
pip install nltk torchtext # additional requirements.
cd examples/text-generation/pplm
```

## PPLM-BoW PPLM模型之词袋法  方法1

### PPLM-BoW 使用方法

```bash
python run_pplm.py -B military --cond_text "The potato" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample
python run_pplm.py -B military_chinese.txt --cond_text "The potato" --length 50 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.03 --window_length 5 --kl_scale 0.01 --gm_scale 0.99 --colorama --sample
```
-B 使用  PPLM-BoW模型, military 生成的文本偏向的关键词，关键词的风格, 变量 BAG_OF_WORDS_ARCHIVE_MAP
--length 生成文本长度50个字
--cond_text 起始单词 
--gamma 梯度归一化时的平方参数
<<<<<<< HEAD
--num_iterations  迭代次数
--num_samples  生成10个段落样本
--stepsize 梯度步长
=======
--num_iterations  扰动次数？
--num_samples  生成10个段落样本
--stepsize 梯度
>>>>>>> d5c81c5b381f39d2c9901bdac34f4ae13a4df56a
--window_length 窗口长度，滑动窗口的长度
--kl_scale KL散度系数
--gm_scale 0.99 
--colorama  字体颜色
--sample  True表示随机获取，False表示贪心方式获取,topk

### PPLM-BoW 优化超参数

1. 增大 `--stepsize` 值可以增强主题控制, 减小可以减弱控制. `--stepsize 0` 0 表示不控制，直接用GPT-2的模型本来的输出

2. 如果生成的语言重复，例如 (For e.g. "science science experiment experiment"), 可以通过一下几个参数调节: </br>
	a) 减小 `--stepsize` </br>
	b) 增大KL散度 `--kl_scale` ，增大KL散度，即KL散度的损失系数， 或者减小 `--gm_scale` (the gm-scaling term) </br>
	c) Add `--grad-length xx` where xx is an (integer <= length, e.g. `--grad-length 30`).</br>


## PPLM-Discrim  PPLM判别器 方法2

### Example command for discriminator based sentiment control

```bash
python run_pplm.py -D sentiment --class_label 2 --cond_text "My dog died" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```

### 优化超参数 for discriminator control

1. 增大 `--stepsize` 值可以增强主题控制, 减小可以减弱控制. `--stepsize 0` 0 表示不控制，直接用GPT-2的模型本来的输出

2. Use `--class_label 3` for negative, and `--class_label 2` for positive

