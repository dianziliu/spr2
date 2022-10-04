import sys
sys.path.append("/home/ljr/code/SPR/")

from data.data import *


import os
from datetime import datetime
from multiprocessing import Pool
from time import sleep

import numpy as np
from model import Simi
from until.data import epoch_process_data_by_split, process_data_by_split,process_data
from until.evulation import (ndcg_evaluations, ndcg_map_evulations,
                             pr_ndcg_evulation_Alls, pred_to_save)

from bpr import BPRArgs,BPR,UniformPairWithoutReplacement
from scipy.sparse import csr_matrix


def evulation_all_datasets_dim(rebuildDataset=False):
    # train_path_format = "run/mlls/ls_dim_train.csv"
    # test_path_format = "run/mlls/ls_dim_test.csv"
    # u_path = "run/m1ls/dim{}_u{}.bin"
    # i_path = "run/m1ls/dim{}_i{}.bin"

    print("start to evulation all datasets dims")
    experiment="Baseline/bpr-master/res/bpr.exp"
    Ks1=[5,10]
    ms=[0]



    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            if not name in ["r3"]:
                continue
            dir=os.path.join("Baseline/data",name)
            train_path_format=os.path.join(dir,"train{}.csv")
            test_path_format=os.path.join(dir,"test{}.csv")
            u_path = os.path.join(dir,"u{}.bin")
            i_path = os.path.join(dir,"i{}.bin")
            
            # 划分数据集
            if not os.path.exists(dir) or rebuildDataset:
                os.mkdir(dir)         
                print("spliting data....")
                dataSplitPool=Pool(10)
                for i in range(5):
                    train_path = train_path_format.format(i)
                    test_path = test_path_format.format(i)
                    dataSplitPool.apply_async(
                        process_data(path, ",", train_path, test_path,0.5)
                    )
                dataSplitPool.close()
                dataSplitPool.join()
            
            # traversing each datset
            # evulation for each dataset
            for dim in [10,20,30,40,50]:

                All_ndcgs = [{5: [], 10: []} for i in range(8)]
                print("dataset",name)
                print("dataset",name,file=f,flush=True)  
                for iter in range(1):
                    train_path = train_path_format.format(iter)
                    test_path = test_path_format.format(iter)
                    data=np.zeros((n_users,n_items))
                    with open(train_path,"r") as tmp_file:
                        tmp_file.readline()
                        for line in tmp_file.readlines():
                            # u,i,r,_=line.split(",")
                            line=line.split(",") 
                            u,i,r=line[0],line[1],line[2]
                            data[int(u),int(i)]=float(r)
                    data=csr_matrix(data)
                    
                    args = BPRArgs()
                    args.learning_rate = 0.01
                    num_factors = dim
                    model = BPR(num_factors,args)
                    model.name="BPR1"
                    sample_negative_items_empirically = True
                    sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
                    num_iters = 10
                    model.train(data,sampler,num_iters)

                    ndcgs=ndcg_evaluations(model,Ks1,test_path)

                    for k in Ks1:
                        All_ndcgs[0][k].append(np.mean(ndcgs[k]))
                    # save_path="res_save/{}_{}_{}_dim{}.csv".format(model.name,name,iter,dim)
                    # pred_to_save(model,test_path,save_path)
                
                print("method {}\tDim={}\tndcg{}:{},ndcg{}:{}".format(
                    "BPR1",dim,5,np.mean(All_ndcgs[0][5]),10,np.mean(All_ndcgs[0][10])
                ),file=f,flush=True)



if __name__=="__main__":
    evulation_all_datasets_dim()
