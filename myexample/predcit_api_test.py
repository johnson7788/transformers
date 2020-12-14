import requests
import base64
import pprint
import json


if __name__ == '__main__':
    url = "http://127.0.0.1:5001/api"
    test_data = ['百亿高活性酵素', 'REPAIRESKIN', '维稳修护', '二裂酵母', '水杨酰植物鞘氨醇', 'BOOSTSKIN', '促进肌肤更新', '木瓜蛋白酶', '轻乙基嗪乙烷磺酸', '*欧莱雅实验室数据', '根据木瓜蛋白酶的添加量和分子量计算得出']
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(r.json())