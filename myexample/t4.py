from transformers import AlbertForMaskedLM, AlbertTokenizer
import torch
tokenizer = AlbertTokenizer.from_pretrained("albert-large-v2")
model = AlbertForMaskedLM.from_pretrained("albert-large-v2")
sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."
input = tokenizer.encode(sequence, return_tensors="pt")
# 被Mask的字符的位置
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
#获取每个位置的logits, [batch_size, seq_length, vocab_size],  torch.Size([1, 28, 30522]), 即最大的可能性
token_logits = model(input)[0]
#只获取被mask处的单词的logits
mask_token_logits = token_logits[0, mask_token_index, :]
# 我们只取前5个可能的结果，从vocab_size众多结果中
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
#打印前5个结果
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))