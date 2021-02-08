# coding=utf-8
# @Date  : 2021/1/27 10:30 上午
# @File  : gen_data.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 翻译的单词的和句子的指标

import datasets


_CITATION = """\
@InProceedings{huggingface:metric,
title = {repair test},
authors={johnson
},
year={2020}
}
"""

_DESCRIPTION = """\
repair metric
"""



_KWARGS_DESCRIPTION = """
这里是描述函数，使用某些分数计算在给出某些参考的情况下预测的效果如何
Args:
    predictions: predictions的score 列表。 每个predictions都应该是一个带有用空格分隔的token的字符串。 
    references: 每个预测的参考列表。 每个引用应该是带有用空格分隔的token的字符串。 
Returns:
    accuracy: first score的描述， 
    another_score: 另外一个score的描述
"""

#自定义一些变量，如果需要的话
# BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


def cal_matthews_corrcoef(references, predictions):
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(references, predictions)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    from sklearn.metrics import f1_score
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "accuracy": acc,
        "f1": f1,
    }


def pearson_and_spearman(preds, labels):
    from scipy.stats import pearsonr, spearmanr
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }

class SmoothMetric(datasets.Metric):
    """metric的描述"""

    def _info(self):
        # 会作为 datasets.MetricInfo 的信息
        return datasets.MetricInfo(
            # 这是将在metric页面上显示的描述。
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # 定义预测和真实标签的格式, 注意预测时的标签格式，一般为int格式, 如果是回归模型为float32
            features=datasets.Features({
                'predictions': datasets.Value("int64"),
                'references': datasets.Value("int64"),
            }),
            homepage="http://metric.homepage",
            #其它介绍信息
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        """如果需要的话，下载外部资源，不需要设置为pass"""
        # TODO: Download external resources if needed
        pass
        # bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
        # self.bad_words = set([w.strip() for w in open(bad_words_path, "r", encoding="utf-8")])

    def _compute(self, predictions, references):
        """
        计算指标返回, score计算, 可以根据不同的name，返回不同的计算方法,例如
        if self.config_name == "cola":
            return {"matthews_correlation": cal_matthews_corrcoef(references, predictions)}
        elif self.config_name == "stsb":
            return pearson_and_spearman(predictions, references)
        elif self.config_name in ["mrpc", "qqp"]:
            return acc_and_f1(predictions, references)
        elif self.config_name in ["sst2", "mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]:
            return {"accuracy": simple_accuracy(predictions, references)}
        Args:
            predictions:  模型的预测值
            references: 真实值
        Returns:

        """
        # 可以返回不同的score, 例如计算准确率
        accuracy = sum(i == j for i, j in zip(predictions, references)) / len(predictions)
        # 计算其它score，如果需要的话, 返回时，也返回second_score就可以了
        # if self.config_name == "max":
        #     second_score = max(abs(len(i) - len(j)) for i, j in zip(predictions, references) if i not in self.bad_words)
        return {
            "accuracy": accuracy,
        }
