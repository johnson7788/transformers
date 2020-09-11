from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

batch_sentences = ["Hello I'm a single sentence",
                   "And another sentence",
                   "And the very very last one"]
batch = tokenizer(batch_sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)