from datasets import list_datasets, load_dataset, list_metrics, load_metric

#列出所有可用数据集
# print(list_datasets())
# weibo_dataset = load_dataset(path='data/weibo_ner.py')
# print(weibo_dataset['train'][0])

# msra_dataset = load_dataset('msra_ner')
msra_dataset = load_dataset('data/msra_ner.py')
print(msra_dataset['train'][:5])

# squad_dataset = load_dataset('squad')
# print(squad_dataset['train'][0])