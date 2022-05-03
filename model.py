# coding=utf-8
import sys
import json
import base64
import numpy as np
import utils
import torch as t

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import quote_plus


# 防止https证书校验不正确
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

API_KEY = 'VCtO5YXcC1FL49bajalGvZNT'

SECRET_KEY = 'a0tQGXwaHvN0maIl4qj3faBOV4Qap0WW'


#IMAGE_RECOGNIZE_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v2/dish"
IMAGE_RECOGNIZE_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient"
#IMAGE_RECOGNIZE_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal"


"""  TOKEN start """
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'


"""
    获取token
"""
def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print(err)
    result_str = result_str.decode()

    result = json.loads(result_str)

    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not 'brain_all_scope' in result['scope'].split(' '):
            print ('please ensure has check the  ability')
            exit()
        return result['access_token']
    
"""
    读取文件
"""
def read_file(image_path):
    f = None
    try:
        f = open(image_path, 'rb')
        return f.read()
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()

"""
    调用远程服务
"""
def request(url, data):
    req = Request(url, data.encode('utf-8'))
    has_error = False
    f = urlopen(req)
    result_str = f.read()
    result_str = result_str.decode()
    return result_str

class baidu_model:
    def __init__(self):
        self.classes = dict()
    
    def predict(self,filename,use_path = True):
        
        token = fetch_token()
        url = IMAGE_RECOGNIZE_URL + "?access_token=" + token
        
        file_content = read_file(filename) if use_path else filename
        response = request(url, urlencode(
            {
                "top_num" : 10
                ,'image': base64.b64encode(file_content)
            }))
        
        result_json = json.loads(response)
        for couple in result_json['result']:
            self.classes.update({couple['name']:couple['score']})
        logits = np.array( [list(self.classes.values())[:10]],dtype=float)
        print(result_json['result'][:3])
        return logits
       

    
    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = utils.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        return loss.flatten()

  
if __name__ == "__main__":
    bdm = baidu_model()
    r = bdm.predict('data/food6.葡萄.jpg')
    print(utils.dense_to_onehot([3],10))


