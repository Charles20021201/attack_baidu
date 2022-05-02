from torch import square
from attack_baidu.attack import square_attack_linf
from model import baidu_model
import torch as t
import numpy as np
import attack
import os
import argparse
import utils
from dataset import dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="choose a image to be the adverserial example" )
    parser.add_argument("--root",type = str,default="./data")
    parser.add_argument("--index",type = int,default=1)
    parser.add_argument('--eps',type = float,default = 0.05)
    parser.add_argument("--n_iter",type = int,default = 20)
    parser.add_argument("--n_cls",type = int,default = 10)
    parser.add_argument("--targeted",type = bool,default=False)
    args = parser.parse_args()
    ds = dataset(args.root)
    dl = enumerate(DataLoader(ds,batch_size=1))
    for i in range(args.index):
        index,(data,label) = next(dl)
    baidu = baidu_model()
    logits_clean = baidu.predict(utils.convert_to_image(data[0]))
    data = data.numpy()
    label = label.numpy()
    y_target = utils.random_classes_except_current(label,args.n_cls) if args.targeted else label
    onehot_label = utils.dense_to_onehot(label,args.n_cls)
    
    if logits_clean.argmax(1) == label :
        n_queries, adv_ex = attack.square_attack_linf(baidu,data,onehot_label,targeted=args.targeted,
            eps = args.eps,n_iters=args.n_iter,p_init=0.05)