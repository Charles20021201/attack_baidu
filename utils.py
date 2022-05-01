import numpy as np
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

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
    y_test_onehot[np.arange(len(y_test)), y_test] = True
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



 
def convert_to_image(img_tensor):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    print(img_tensor.shape)
    img_tensor = img_tensor.clone()  
    img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
    img_tensor = img_tensor.detach().numpy()*255
    
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()
    img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    return img