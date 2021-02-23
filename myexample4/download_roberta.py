from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer
import os
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("clue/roberta_chinese_base")
model.save_pretrained('myroberta')
tokenizer.save_pretrained('myroberta')
os.remove("myroberta/special_tokens_map.json")
os.remove("myroberta/tokenizer_config.json")
# os.system("mv deberta-base ../")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# model.save_pretrained('bert_model_uncased')
# tokenizer.save_pretrained('bert_model_uncased')
