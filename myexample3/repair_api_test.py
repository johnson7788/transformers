import requests
import base64
import pprint
import json


if __name__ == '__main__':
    url = "http://127.0.0.1:6666/api"
    test_data = [['如果高资源语言没有经过足够的培训，则该模型将失效',[('underﬁt','失效',['欠拟合']),('train','培训',['训练']),('high-resource','高资源',['high-resource'])]],
                 ['读出层通过获取子图的隐藏表示的总和/平均值来总结最终的图表示', [('mean', '值', ['值']), ('graph', '图', ['图','图形'])]]
                 ]
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(r.json())