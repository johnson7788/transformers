from nlp import Dataset
def read_ner_txt(file_path):
    """
    txt文件内容的格式是一行有2列，第一列是一个字，第二列是对应的实体类别
    每个句子都是以空行分隔

    Args:
        file_path: 文件路径

    Returns: Dict[list]
    {'ner_tags':[[a,b,c,],...,[a,b,c]], 'tokens':[[x,y,z,],...,[x,y,z]]}
    """
    tokens = []
    ner_tags = []
    with open(file_path,'r') as f:
        sentence = []
        sentence_tag = []
        for line in f:
            if line != '\n':
                line_split = line.split('\t')
                if len(line_split) == 2:
                    one_token, one_tag = line_split
                    # 去掉右侧的换行
                    one_tag = one_tag.rstrip('\n')
                    sentence.append(one_token)
                    sentence_tag.append(one_tag)
                else:
                    print(f"这一行出现问题{line}，不是2个字段")
            else:
                #如果是空行，那么说明是下一句了,需要把sentence和 sentence_tag 加入到总的tokens和ner_tags中，然后重置
                if sentence and sentence_tag:
                    tokens.append(sentence)
                    ner_tags.append(sentence_tag)
                    sentence = []
                    sentence_tag = []
    if len(tokens) != len(ner_tags):
        print(f'tokens 和ner_tags的长度不相等，读取的文件有问题,请检查')
        result = {'tokens': [], 'ner_tags': []}
    else:
        result = {'tokens':tokens, 'ner_tags':ner_tags}
    return result

if __name__ == '__main__':
    dev_file = "dataset/cosmetics/dev.txt"
    train_file = "dataset/cosmetics/train.txt"
    test_file = "dataset/cosmetics/test.txt"
    test_dict = read_ner_txt(test_file)
    dataset = Dataset.from_dict(test_dict)
    print(dataset)