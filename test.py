
# encoding:utf-8

import requests
import base64

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
    
#request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal"

access_token =  fetch_token()
request_url = IMAGE_RECOGNIZE_URL + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
import time
def get_result(filename):
    f = open(filename, 'rb')
    img = base64.b64encode(f.read())
    params = {"image":img,'top_num':10}
    response = requests.post(request_url, data=params, headers=headers)
    time.sleep(0.5)
    print ( list(response.json()['result'])[:3])
if __name__ == '__main__':
    f1 = 'source.jpg'
    f2 = './convert_process/Image.315.jpg'
    for f in (f1,f2):
        get_result(f)