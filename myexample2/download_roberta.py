from transformers import BertTokenizer, AlbertForMaskedLM
import os
pretrained = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(pretrained)
model = AlbertForMaskedLM.from_pretrained(pretrained)

model.save_pretrained('roberta_model')
tokenizer.save_pretrained('roberta_model')
# os.remove("albert_model/special_tokens_map.json")
# os.remove("albert_model/tokenizer_config.json")
# os.system("mv albert_model ../")
