import connection
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
    parser.add_argument('--eps',type = float,default = 0.3)
    parser.add_argument("--n_iter",type = int,default = 100)
    parser.add_argument("--n_cls",type = int,default = 10)
    args = parser.parse_args()
    ds = dataset(args.root)
    dl = enumerate(DataLoader(ds,batch_size=1))
    for i in range(args.index):
        index,(data,label) = next(dl)
    
    #y_target = utils.random_classes_except_current(label,args.n_cls)
    #y_target_onehot = utils.dense_to_onehot(y_target,n_cls = args.n_cls)
    #n_queries, adv_ex = attack.square_attack_linf()