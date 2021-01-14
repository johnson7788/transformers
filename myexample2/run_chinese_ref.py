import argparse
import json
from typing import List

from ltp import LTP
from transformers import BertTokenizer


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word: str):
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return 0
    return 1


def get_chinese_word(tokens: List[str]):
    word_set = set()

    for token in tokens:
        chinese_word = len(token) > 1 and is_chinese(token)
        if chinese_word:
            word_set.add(token)
    word_list = list(word_set)
    return word_list


def add_sub_symbol(bert_tokens: List[str], chinese_word_set: set()):
    """
    根据LTP对一个句子的分词，把bert的token转换成Whole word的形式，
    Args:
        bert_tokens: ['[CLS]', '老', '用', '户', '了', '，', '一', '直', '在', '用', '满', '婷', '，', '感', '觉', '对', '控', '痘', '控', '油', '效', '果', '挺', '好', '的', '[SEP]']
        chinese_word_set: 中文的分词结果， ['效果', '用户', '感觉', '一直']

    Returns:
        ['[CLS]', '老', '用', '##户', '了', '，', '一', '##直', '在', '用', '满', '婷', '，', '感', '##觉', '对', '控', '痘', '控', '油', '效', '##果', '挺', '好', '的', '[SEP]']
    """
    if not chinese_word_set:
        return bert_tokens
    max_word_len = max([len(w) for w in chinese_word_set])

    bert_word = bert_tokens
    start, end = 0, len(bert_word)
    while start < end:
        single_word = True
        if is_chinese(bert_word[start]):
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                whole_word = "".join(bert_word[start : start + i])
                if whole_word in chinese_word_set:
                    for j in range(start + 1, start + i):
                        bert_word[j] = "##" + bert_word[j]
                    start = start + i
                    single_word = False
                    break
        if single_word:
            start += 1
    return bert_word


def prepare_ref(lines: List[str], ltp_tokenizer: LTP, bert_tokenizer: BertTokenizer):
    """
    Args:
        lines:  每行一个中文段落，
        ltp_tokenizer: ltp的tokenizer处理器
        bert_tokenizer:  bert的tokenizer处理器
    Returns:

    """
    ltp_res = []
    # 每次处理100行
    print(f"开始用ltp模型进行分词处理...")
    for i in range(0, len(lines), 100):
        #调用ltp进行分词
        res = ltp_tokenizer.seg(lines[i : i + 100])[0]
        #过滤出分词后都是中文的部分
        res = [get_chinese_word(r) for r in res]
        #加到ltp_res
        ltp_res.extend(res)
    assert len(ltp_res) == len(lines)
    # eg: ltp_res中的文本处理的结果 [ ['效果', '一直', '用户', '感觉'],....]
    #bert也进行tokenizer， 每次处理100行
    print(f"开始用bert tokenizer模型进行token处理...")
    bert_res = []
    for i in range(0, len(lines), 100):
        res = bert_tokenizer(lines[i : i + 100], add_special_tokens=True, truncation=True, max_length=512)
        bert_res.extend(res["input_ids"])
    # eg: bert_res [ [101, 5439, 4500, 2787, 749, 8024, 671, 4684, 1762, 4500, 4007, 2051, 8024, 2697, 6230, 2190, 2971, 4576, 2971, 3779, 3126, 3362, 2923, 1962, 4638, 102]...]
    #确保行数相同
    print(f"开始生成对应关系")
    assert len(bert_res) == len(lines)
    print_num = 5
    ref_ids = []
    for input_ids, chinese_word in zip(bert_res, ltp_res):
        input_tokens = []
        for id in input_ids:
            token = bert_tokenizer._convert_id_to_token(id)
            input_tokens.append(token)
        # eg : ['[CLS]', '古', '##龙', '洗', '发', '##水', '，', '洗', '完', '头', '##发', '不', '干', '##燥', '、', '也', '不', '容', '##易', '油', '、', '不', '痒', '，', '味', '##道', '持', '##久', '，', '非', '##常', '柔', '##顺', '，', '而', '##且', '泡', '##泡', '很', '容', '##易', '冲', '##洗', '干', '##净', '泡', '##沫', '非', '##常', '细', '##腻', '，', '洗', '后', '头', '##发', '很', '滑', '很', '顺', '，', '洗', '了', '之', '##后', '就', '头', '##发', '很', '蓬', '##松', '，', '很', '香', '，', '而', '##且', '我', '洗', '了', '是', '没', '##有', '头', '##皮', '##屑', '的', '[SEP]']
        input_tokens = add_sub_symbol(input_tokens, chinese_word)
        ref_id = []
        # 我们只保存以##开头的中文子词的位置，这意味着它是全词的一部分。
        for i, token in enumerate(input_tokens):
            if token[:2] == "##":
                clean_token = token[2:]
                # 只保存中文子词的后半部分，把和bert的对应关系，保存到ref_id中，ref_id是这个句子的所有子词的后半部分映射
                if len(clean_token) == 1 and _is_chinese_char(ord(clean_token)):
                    ref_id.append(i)
        #打印前5个示例
        if print_num >0:
            example_num = 5 - print_num
            print(f"第{example_num}个样本是: {lines[example_num]}")
            print(f"第{example_num}个样本的ltp分词后结果: {ltp_res[example_num]}")
            print(f"第{example_num}个样本的bert toknizer后结果: {bert_res[example_num]}")
            print(f"第{example_num}个样本的bert toknizer被ltp的全词处理后的结果: {input_tokens}")
            print(f"第{example_num}个样本的bert的token对应的子词的后半部分的位置的最终的ref_id: {ref_id}")
            print()
            print_num -=1
        ref_ids.append(ref_id)
    #判断每个句子的子词的映射关系都保存了
    assert len(ref_ids) == len(bert_res)

    return ref_ids


def main(args):
    # For Chinese (Ro)Bert, the best result is from : RoBERTa-wwm-ext (https://github.com/ymcui/Chinese-BERT-wwm)
    # 如果要微调这些模型，则必须使用相同的tokenizer  : LTP (https://github.com/HIT-SCIR/ltp)
    with open(args.file_name, "r", encoding="utf-8") as f:
        data = f.readlines()
    print(f'开始处理数据,共有{len(data)}条')
    data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]  # avoid delimiter like '\u2029'
    print(f"开始加载ltp和bert的tokenizer模型")
    ltp_tokenizer = LTP(path=args.ltp)  # faster in GPU device
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert)
    #准备映射关系
    ref_ids = prepare_ref(data, ltp_tokenizer, bert_tokenizer)
    #保存映射关系
    with open(args.save_path, "w", encoding="utf-8") as f:
        data = [json.dumps(ref) + "\n" for ref in ref_ids]
        f.writelines(data)
    print(f"保存所有{len(data)}条数据的映射关系到文件{args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备中文参考")
    parser.add_argument(
        "--file_name",
        type=str,
        default="data/demo.txt",
        help="需要处理的文件，例如训练数据, 每行一个文本",
    )
    parser.add_argument(
        "--ltp", type=str, default="small", help="LTP tokenizer模型, 使用小型模型，那么，写成small即可"
    )
    parser.add_argument("--bert", type=str, default="bert-base-chinese", help="Bert的tokenizer模型")
    parser.add_argument("--save_path", type=str, default="data/ref.txt", help="输出保存参考的位置")

    args = parser.parse_args()
    main(args)
