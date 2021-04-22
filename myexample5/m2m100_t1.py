#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/4/22 4:35 下午
# @File  : m2m100_t1.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
text_to_translate = "Life is like a box of chocolates"
model_inputs = tokenizer(text_to_translate, return_tensors='pt')
#英语翻译到汉语
gen_tokens = model.generate( **model_inputs, forced_bos_token_id=tokenizer.get_lang_id("zh"))
print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))