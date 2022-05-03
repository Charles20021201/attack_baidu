from torch import square
from attack import square_attack_linf
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
    parser.add_argument("--atta_method",type=str,default='linf')
    parser.add_argument("--target_class",type = str,default = 'fruits')
    parser.add_argument("--target_part",type= str,default='gender')
    args = parser.parse_args()
    args.root = "./{}_data".format(args.target_class)
    ds = dataset(args.root)
    dl = enumerate(DataLoader(ds,batch_size=1))
    for i in range(args.index):
        index,(data,label) = next(dl)
    baidu = baidu_model()
    logits_clean = baidu.predict(utils.convert_to_image(data[0]),args.target_class,args.target_part)
    data = data.numpy()
    label = label.numpy()
    args.n_cls = 2 if args.target_class == 'face' else 10
    y_target = utils.random_classes_except_current(label,args.n_cls) if args.targeted else label
    onehot_label = utils.dense_to_onehot(label,args.n_cls)
    attack_way = attack.square_attack_linf if args.atta_method == 'linf' else attack.square_attack_l2
    p_init = 0.05 if args.atta_method == 'linf' else 0.1
    if logits_clean.argmax(1) == label :
        n_queries, adv_ex = attack_way(baidu,data,onehot_label,targeted=args.targeted,
            eps = args.eps,n_iters=args.n_iter,p_init = p_init,target_class=args.target_class,target_part = args.target_part)