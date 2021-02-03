#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/3 3:12 下午
# @File  : deberta_MLM.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

def base_deberta():
    from transformers import DebertaTokenizer, DebertaModel
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    model = DebertaModel.from_pretrained('microsoft/deberta-base')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    #输出最后一层的隐藏层状态
    print(last_hidden_states)

def base_lm():
    from transformers import DebertaTokenizer, DebertaForMaskedLM
    import torch
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    model = DebertaForMaskedLM.from_pretrained('microsoft/deberta-base')
    inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
    labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    print(loss)
    print(logits)

def sequence_classify():
    """
     文件src/transformers/models/deberta/modeling_deberta.py的1169行有一些问题，所以会报错，维度不匹配的错误, 在BERT上是没有此种错误的
     RuntimeError: Index tensor must have the same number of dimensions as input tensor
     labels = torch.gather(labels, 0, label_index.view(-1))
    Returns:

    """
    from transformers import DebertaTokenizer, DebertaForSequenceClassification
    import torch
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  #假设标签为1， 这里以一个样本，Batch size 也是 1
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    print(loss)
    print(logits)

def bert_test():
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits

def base_token():
    from transformers import DebertaTokenizer, DebertaForTokenClassification
    import torch
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    model = DebertaForTokenClassification.from_pretrained('microsoft/deberta-base')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # 假设每个token的label都是1，默认的num_labels是2
    labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    print(loss)
    print(logits)

def base_qa():
    """
    问答的跨度任务, 损失是span开始的位置的损失+span预测结束位置的损失之和除以2
    Returns:
    """
    from transformers import DebertaTokenizer, DebertaForQuestionAnswering
    import torch
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    model = DebertaForQuestionAnswering.from_pretrained('microsoft/deberta-base')
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    inputs = tokenizer(question, text, return_tensors='pt')
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])
    outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    print(loss)
    print(start_scores)
    print(end_scores)

if __name__ == '__main__':
    # base_deberta()
    # base_lm()
    # sequence_classify()
    # bert_test()
    # base_token()
    base_qa()