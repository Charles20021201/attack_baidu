import numpy as np
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
class Logger:
    def __init__(self, path):
        self.path = path
        if path != '':
            folder = '/'.join(path.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

    def print(self, message):
        print(message)
        if self.path != '':
            with open(self.path, 'a') as f:
                f.write(message + '\n')
                f.flush()


def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), 0] = True
    
    return y_test_onehot


def random_classes_except_current(y_test, n_cls):
    y_test_new = np.zeros_like(y_test)
    for i_img in range(y_test.shape[0]):
        lst_classes = list(range(n_cls))
        lst_classes.remove(y_test[i_img])

        y_test_new[i_img] = np.random.choice(lst_classes)
    return y_test_new


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def convert_to_image(img_tensor,path = None):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.clone().numpy()
    img_tensor = img_tensor.transpose(2,1,0).transpose(1,0,2)  
    img_tensor = img_tensor*255
    img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    if not path:
        path = 'source.jpg'
        plt.imshow(img)
        plt.savefig(path)
    else:
        plt.imshow(img)
        plt.savefig(path)  
    return path