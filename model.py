# coding=utf-8
import sys
import json
import base64
import numpy as np
import utils
import torch as t
import requests

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
IMAGE_RECOGNIZE_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient"
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'

def fetch_infor(target_class):
    def fetch_URL( ):
        global API_KEY,SECRET_KEY
        if target_class == 'fruits':
            return "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient"
        if target_class == 'animal':
            return "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal"
        if target_class == 'dish':
            return "https://aip.baidubce.com/rest/2.0/image-classify/v2/dish"
        if target_class == 'face':
            API_KEY = 'EbI2GRSR1XnkDGbqvapMVAw3'
            SECRET_KEY = 'w9EVqB8EGgEkpPEkb0SyMONcDZb4ZGGO'
            return 'https://aip.baidubce.com/rest/2.0/face/v3/detect'
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
    return fetch_URL(),fetch_token()

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
    
    def predict(self,filename,target_class = 'fruits' ,target_part = 'gender'):
        IMAGE_RECOGNIZE_URL,token = fetch_infor(target_class)
        url = IMAGE_RECOGNIZE_URL + "?access_token=" + token
        headers = {'content-type': 'application/json'}
        file_content = read_file(filename)
        img = base64.b64encode(file_content )
        if target_class != 'face':
            response = request(url, urlencode({
                    "top_num" : 10
                    ,'image': img
                }))
            result_json = json.loads(response)
            for couple in result_json['result']:
                self.classes.update({couple['name']:couple['probability']})
            logits = np.array( [list(self.classes.values())[:10]],dtype=float)
            print(result_json['result'][:3])
        else:
            params = {"image": img,"image_type":"BASE64","face_field":"age,emotion,gender,mask,glasses,eye_status,quality,face_shape,face_type"}
            result = requests.post(url, data=params, headers=headers).json()
            if 'error_msg' in result and result['error_msg'] =='pic not has face':
                prob = 0.0
            else:
                features = result['result']['face_list'][0]
                if target_part in ['gender','face_shape','glasses','mask','emotion'] :
                    print(features[target_part])
                    if not self.classes or list(self.classes.keys())[0] == features[target_part]['type']:
                        #print(features[target_part]['type'])
                        self.classes.update({features[target_part]['type']:features[target_part]['probability']})
                        prob = list(self.classes.values())[0]
                    elif features[target_part]['type'] in ['pouty'] :
                       
                        self.classes.update({features[target_part]['type']:features[target_part]['probability']})
                        prob = list(self.classes.values())[0]
                    else:
                        prob = 0.0
                if target_part == 'eye':
                    for couple in features['eye_status'].items():
                        self.classes.update({couple[0]:couple[1]})
                    prob = max(list(self.classes.values())[0],0)
                if target_part == 'age':
                    if not self.classes :
                        self.classes.update({'age':features['age']})
                    age = features['age']
                    prob = 0.05 * (20-abs(self.classes['age']-age))
            sup_prob = 1-prob if target_part not in ['face_shape','emotion'] else 0.01
            logits = np.array( [[prob, sup_prob]],dtype = float)
            print('has_feature : {}'.format(prob))
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
    result = bdm.predict('./face_data/Image0.男性.jpg','face')
    print(result)


