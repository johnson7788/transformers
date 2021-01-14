from transformers import BertTokenizer, AutoModelForMaskedLM, BertModel, RobertaModel
import os

def robera():
    pretrained = 'hfl/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = AutoModelForMaskedLM.from_pretrained(pretrained)

    model.save_pretrained('roberta_model')
    tokenizer.save_pretrained('roberta_model')
    # os.remove("albert_model/special_tokens_map.json")
    # os.remove("albert_model/tokenizer_config.json")
    # os.system("mv albert_model ../")

def rbt3():
    """
    RBT3 3层RoBERTa-wwm-ext-base
    RBTL3（3层RoBERTa-wwm-ext-base/large)
    Returns:

    """
    pretrained = "hfl/rbt3"
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = RobertaModel.from_pretrained(pretrained)

    model.save_pretrained('rbt3')
    tokenizer.save_pretrained('rbt3')

if __name__ == '__main__':
    robera()
    # rbt3()