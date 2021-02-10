import requests
import base64
import pprint
import json


if __name__ == '__main__':
    url = "http://127.0.0.1:5001/api"
    test_data = [['如果高资源语言没有经过足够的培训，则该模型将失效',[('underﬁt','失效'),('train','培训'),('high-resource','高资源')]],['为了尽早停止，我们每200个步骤保存一个检查点，并在验证集上选择性能最高的检查点',[('step','个步骤'),('Stop early','尽早停止'),('checkpoint','检查点')]]]
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(r.json())