from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir="cache_deberta")
model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-base", cache_dir="cache_deberta")
model.save_pretrained('deberta-base')
tokenizer.save_pretrained('deberta-base')
# os.remove("deberta-base/special_tokens_map.json")
# os.remove("deberta-base/tokenizer_config.json")
# os.system("mv deberta-base ../")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# model.save_pretrained('bert_model_uncased')
# tokenizer.save_pretrained('bert_model_uncased')
