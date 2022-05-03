import os
import torch as t
import torchvision.transforms as T
from PIL import Image
import numpy as py
from torch.utils import data


class dataset(data.Dataset):
    def __init__(self,root):
        '''
        获取所有图片
        '''
        imgs = [os.path.join(root,image) for image in os.listdir(root)]
        self.imgs = imgs
        self.transform = T.Compose([T.CenterCrop([1200,1000]),T.ToTensor()])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.transform(data)
        return data,0
    

    def __len__(self):
        '''
        返回数据集中图片的个数
        '''
        return len(self.imgs)