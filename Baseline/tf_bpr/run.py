import os
import sys

sys.path.append(".")
from datetime import datetime
from multiprocessing import Pool
from time import sleep

import numpy as np
from data.data import *
from model import Simi
from until.data import process_data, process_data_by_split
from until.evulation import (pred2save,ndcg_evulations_pred)

from Baseline.tf_bpr.bpr import get_model,bpr_train
from until.logger import logger_config

dim=10

def evulation_all_datasets(rebuildDataset=False):
    # log_file="spr_logall.log"
    experiment = "Baseline/tf_bpr/bpr_exps"
    model_name="TF_BPR"
    Ks1 = [5, 10]
    Ks2 = [10, 20]
    # 1. 打开日志记录
    logger=logger_config(experiment,"logger")
    for data_name, data_path, n_users, n_items in zip(dataset, paths, n_userss, n_itmess):
        
        dir=os.path.join("run",data_name)
        dir2=os.path.join("Baseline/tf_bpr/res",data_name)
        if not os.path.exists(dir2):
            os.mkdir(dir2)
        train_path_format = os.path.join(dir,"train{}.csv")
        test_path_format  = os.path.join(dir,"test{}.csv")
        save_path_format  = os.path.join(dir2,"tf_bpr_pred{}.csv")

        # build_data(rebuildDataset, path, n_items, dir, train_path_format, test_path_format)            
        logger.info("dataset:{}".format(data_name))
        
        # train(data_name, n_users, n_items, train_path_format, test_path_format, save_path_format)
            
        evulation(data_name,model_name,Ks1,logger,save_path_format)


def evulation(data_name,model_name,Ks,logger,save_path_format,fold=5):
    All_ndcgs = {5: [], 10: []}

    # 计算结果
    for i in range(fold):
        print("start to {}th run:{}".format(i+1, data_name))
        save_path = save_path_format.format(i)
        ndcgs = ndcg_evulations_pred(model_name, [5, 10], save_path)
        for k in Ks:
            All_ndcgs[k].append(np.mean(ndcgs[k]))
    # 打印结果
    logger.info("method {}\tndcg{}:{}\tndcg{}:{}".format(
            model_name, 5, np.mean(All_ndcgs[5]), 10, np.mean(All_ndcgs[10])
        ))
    # print("method {},\nndcg{}:{},ndcg{}:{}".format(
    #     model_name, 5, np.mean(All_ndcgs[5]), 10, np.mean(All_ndcgs[10])
    # ), file=log_file, flush=True)


def train(data_name, n_users, n_items, train_path_format, test_path_format, save_path_format,fold=5):
    for i in range(fold):
        print("start to {}th run:{}".format(i+1, data_name))
        train_path = train_path_format.format(i)
        test_path = test_path_format.format(i)
        save_path = save_path_format.format(i)
        model = get_model(n_users, n_items, dim)
        model = bpr_train(model, train_path)
        pred2save(model, test_path, save_path)
    return 


def build_data(rebuildDataset, path, n_items, dir, train_path_format, test_path_format):
    # 划分数据集
    if (not os.path.exists(dir)) or rebuildDataset:
        os.mkdir(dir)         
        print("spliting data....")
        dataSplitPool=Pool(5)
        for i in range(5):
            train_path = train_path_format.format(i)
            test_path = test_path_format.format(i)
            dataSplitPool.apply_async(
                        process_data_by_split,(path,",",n_items,train_path,test_path,0.5,4)
                    )
        dataSplitPool.close()
        dataSplitPool.join()
        print("split data finished")

if __name__ == "__main__":
    evulation_all_datasets()
