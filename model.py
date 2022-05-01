# coding=utf-8
import sys
import json
import base64

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
    else:
        print ('please overwrite the correct API_KEY and SECRET_KEY')
        exit()

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
    try:
        f = urlopen(req)
        result_str = f.read()
        result_str = result_str.decode()
        return result_str
    except  URLError as err:
        print(err)

class baidu_model:
    def predict(self,filename,use_path = True):
        # 获取图片
        token = fetch_token()
        url = IMAGE_RECOGNIZE_URL + "?access_token=" + token
        #print(filename)
        file_content = read_file(filename) if use_path else filename
        
        response = request(url, urlencode(
            {
                'image': base64.b64encode(file_content),
                'top_num': 10
            }))
        result_json = json.loads(response)
        #print(result_json['result'],result_json['result_num'],result_json.items())
        #logits = [couple for couple in result_json]
        result = [couple['score'] for couple in result_json['result'] ]
        return result
    
if __name__ == "__main__":
    bdm = baidu_model()
    r = bdm.predict('./data/food6.葡萄.jpg')
    print(r)


