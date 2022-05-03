# encoding:utf-8
import requests
import base64
import sys
import json
import numpy as np
import torch as t
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import quote_plus
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
API_KEY = 'EbI2GRSR1XnkDGbqvapMVAw3'
SECRET_KEY = 'w9EVqB8EGgEkpPEkb0SyMONcDZb4ZGGO'
IMAGE_RECOGNIZE_URL = 'https://aip.baidubce.com/rest/2.0/face/v3/detect'
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'

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

access_token =  fetch_token()
request_url = IMAGE_RECOGNIZE_URL + "?access_token=" + access_token
headers = {'content-type': 'application/json'}

def get_result(filename):
    f = open(filename, 'rb')
    img = base64.b64encode(f.read())
    params = {"image": img,"image_type":"BASE64","face_field":"age,emotion,gender,mask,glasses,eye_status,quality,face_shape,face_type"}
    response = requests.post(request_url, data=params, headers=headers)
    return response.json()

if __name__ == '__main__':
    #在此处修改图片的地址即可
    file = './data//source.jpg'
    res = get_result(file)['result']['face_list'][0]
    for feature in ['gender','eye_status','mask','face_shape','age','glasses','emotion']:
        print(res[feature])
    for file in ['./data/source copy.jpg']:
        res = get_result(file)['result']['face_list'][0]
        for feature in ['gender','eye_status','mask','face_shape','age','glasses','emotion']:
            print(res[feature])
    
    

