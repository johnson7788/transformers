import pandas as pd
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
    print(f"样本数{len(tokens)}")
    token_length = [len(t) for t in tokens]
    print(f"最长样本{max(token_length)}")
    big_then = sum(1 for l in token_length if l>30)
    print(f"大于长度30的样本个数{big_then}")
    return result
def gen_excel(text, labels, predicts):
    """
    Args:
        text: 空格分隔的文本组成的列表
        labels: 真实标签
        predicts: 预测标签

    Returns:

    """
    def towords(text, nertag):
        """
        把label或者pridict还原成单词
        返回一个列表，列表中子列表，子列表中包含keywrod
        Args:
            text:
            nertag:

        Returns:
        """
        eff_words = []
        com_words = []
        for line, tag in zip(text,nertag):
            #每行的功效词和成分词
            line_eff_words = []
            line_com_words = []
            tag_list = tag.split(" ")
            line_list = line.split(" ")
            #一个单词，把每个字都加进来，临时存储
            one_words = ""
            for idx, every_tag in enumerate(tag_list):
                #对应的字
                word = line_list[idx]
                if every_tag == "O":
                    if one_words and idx !=0:
                        # 如果是O，并且one_words由内容，需要判断上一个词是什么词，并且判断one_words是什么词，是EFF还是COM类型
                        # idx等于0时肯定不用
                        last_tag = tag_list[idx - 1]
                        if last_tag.endswith("COM"):
                            #说明one_words里面是成分类,否则是成分类，因为我们就2种类别
                            line_com_words.append(one_words)
                        else:
                            line_eff_words.append(one_words)
                        #重置one_words
                        one_words = ""
                #说明2个关键词挨着了，需要把旧词保存
                if every_tag == "B-COM":
                    #把旧的保存
                    if one_words:
                        #把字和功效都加进去
                        line_com_words.append(one_words)
                    #加入新的
                    one_words = word
                if every_tag == "I-COM":
                    one_words += word
                #处理EFF词
                if every_tag == "B-EFF":
                    #把旧的保存
                    if one_words:
                        #把字和功效都加进去
                        line_eff_words.append(one_words)
                    #加入新的
                    one_words = word
                if every_tag == "I-EFF":
                    one_words += word
            #处理最后一个词
            if one_words:
                # 把词加进去， 判断最后一个tag是什么
                last_tag = tag_list[-1]
                if last_tag.endswith("COM"):
                    # 说明one_words里面是成分类,否则是成分类，因为我们就2种类别
                    line_com_words.append(one_words)
                else:
                    line_eff_words.append(one_words)
            #每行的结果，加入列表
            eff_words.append(" ".join(line_eff_words))
            com_words.append(" ".join(line_com_words))
        return eff_words, com_words

    predict_eff_words, predict_com_words = towords(text,predicts)
    label_eff_words, label_com_words = towords(text,labels)
    print("各个长度")
    print(len(text),len(labels),len(predicts),len(label_com_words),len(label_eff_words),len(predict_com_words),len(predict_eff_words))
    df = pd.DataFrame({'text':text, 'labels':labels, 'predicts':predicts, '真实标签的功效词':label_eff_words,
                       '预测标签的功效词':predict_eff_words,'真实标签的成分词':label_com_words,'预测标签成分词':predict_com_words})
    writer = pd.ExcelWriter("output.xlsx")
    df.to_excel(writer)
    writer.save()

def read_prediction(file="cosmetic_ner/test_predictions.txt"):
    lines = []
    with open(file) as f:
        for line in f:
            if line != '\n':
                lines.append(line.strip())
    return lines
if __name__ == '__main__':
    res = read_ner_txt(file_path="dataset/cosmetic/test.txt")
    tokens = res['tokens']
    tags = res['ner_tags']
    text = [" ".join(t) for t in tokens]
    labels = [" ".join(t) for t in tags]
    predicts = read_prediction()
    gen_excel(text,labels,predicts)