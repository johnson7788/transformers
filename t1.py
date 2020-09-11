# from transformers import pipeline
# nlp = pipeline("sentiment-analysis")
# result = nlp("I hate you")[0]
# print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
# result = nlp("I love you")[0]
# print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

from transformers import ReformerTokenizer, ReformerModel
import torch
tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
model = ReformerModel.from_pretrained('google/reformer-crime-and-punishment', return_dict=True)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)