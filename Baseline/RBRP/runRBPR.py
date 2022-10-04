import sys

sys.path.append("/home/ljr/code/SPR")
import os
from time import sleep
from datetime import datetime
import numpy as np
import pandas as pd

# from model import Simi
from until.data import process_data_by_split,process_data
from until.evulation import (ndcg_evaluations, ndcg_map_evulations_normal,pred_to_save,
                             pr_ndcg_evulation_Alls)
from RBPR import *
from data.data import *

from multiprocessing import Pool

# paths = ["data/Ml100Krating.csv",
#          "data/ML1Mratings.csv",
#          "data/YahooR3.csv",
#          "data/YahooR4.csv"]
# n_userss=[6400,6400,16000,8000]
# n_itmess=[4000,4000,1024,12000]
# dataset=["ml100k","ml1m","r3","r4"]


train_path = "Baseline/res/train.csv"
test_path = "Baseline/res/test.csv"

# paths = ["data/ML10Mratings.csv",
#         "data/Ml100Krating.csv",
#          "data/ML1Mratings.csv",
#          "data/YahooR3.csv",
#          "data/YahooR4.csv"]

# # ML100K,ML1M,YahooR3,YahooR4
# n_userss=[80000,1000,6400,16000,8000]
# n_itmess=[80000,2000,4000,1024,12000]

# dataset=["ml10m","ml100k","ml1m","r3","r4"]

dim=10
lr=0.007
rg=0.05
iter=20

 
def transdata(old_path,new_path):
    df=pd.read_csv(old_path)
    df=df[["userId","movieId","rating"]]
    df.to_csv(new_path,sep="\t",index=False,header=False)



def evulation_all_datasets():
    
    print("start to evulation_all_datasets RBPR...")
    Dims=[10,20,30,40,50,60,80,100]
    experiment="Baseline/res/exps"
    resultSaveFile="Baseline/res/tmp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    Method=["RBRP"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")       
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            if name not in ["r4"]:
                continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            for i in range(5):
                # process_data_by_split(path,",",n_items,train_path,test_path,n_ng=4)
                process_data(path, ",", train_path, test_path)
                new_train_path="Baseline/res/RBPR_train.csv"
                transdata(train_path,new_train_path)
                #model traing...
                model = RBPR(new_train_path, resultSaveFile, '\t', dim, iter)
                model.train_rbpr()
                model.name=Method[0]
                # 进行评价
                ndcgs,maps=ndcg_map_evulations_normal(model,Ks1,Ks2,train_path,test_path,n_items)
                
                for k in Ks1:
                    All_ndcgs[0][k].append(np.mean(ndcgs[k]))
                for k in Ks2:
                    All_maps[0][k].append(np.mean(maps[k]))

            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)



def evulation_all_datasets_dims():
    
    print("start to evulation_all_datasets_dims RBPR...")
    Dims=[10,20,30,40,50,60,80,100]
    experiment="Baseline/res/dim_exps"
    resultSaveFile="Baseline/res/tmp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    Method=["RBRP"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")       
        for name,path,n_users,n_items in zip(dataset[1:],paths[1:],n_userss[1:],n_itmess[1:]):
            # if name not in ["r4"]:
            #     continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: dict(), 10: dict()} for i in range(8)]
            All_maps = [{10: dict(), 20: dict()} for i in range(8)]

            for dim  in Dims:
                for k in Ks1:
                    All_ndcgs[0][k][dim]=[]
                for k in Ks2:
                    All_maps[0][k][dim]=[]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            for i in range(5):
                # process_data_by_split(path,",",n_items,train_path,test_path,n_ng=4)
                process_data(path, ",", train_path, test_path)
                new_train_path="Baseline/res/RBPR_train.csv"
                transdata(train_path,new_train_path)
                #model traing...
                for dim in Dims:
                    model = RBPR(new_train_path, resultSaveFile, '\t', dim, iter)
                    model.train_rbpr()
                    model.name=Method[0]
                    # 进行评价
                    ndcgs,maps=ndcg_map_evulations_normal(model,Ks1,Ks2,train_path,test_path,n_items)
                    
                    for k in Ks1:
                        All_ndcgs[0][k][dim].append(np.mean(ndcgs[k]))
                    for k in Ks2:
                        All_maps[0][k][dim].append(np.mean(maps[k]))

            for method in ms:
                for dim in Dims: 
                    print("dim={}\tmethod {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                        dim,Method[method],5,np.mean(All_ndcgs[method][5][dim]),10,np.mean(All_ndcgs[method][10][dim]),
                        10,np.mean(All_maps[method][10][dim]),20,np.mean(All_maps[method][20][dim])
                    ),file=f,flush=True)
   


def evulation_ml10m():
    
    print("start to evulation_ml10m_datasets RBPR...")
    Dims=[10,20,30,40,50,60,80,100]
    experiment="Baseline/res/exps"
    resultSaveFile="Baseline/res/tmp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    Method=["RBRP"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")       
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            # if name not in ["r4"]:
            #     continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            train_path_fmt=os.path.join("run",name,"train{}.csv")
            test_path_fmt=os.path.join("run",name,"test{}.csv")
            for i in range(5):
                # process_data_by_split(path,",",n_items,train_path,test_path,n_ng=4)

                train_path=train_path_fmt.format(i)
                test_path=test_path_fmt.format(i)
                
                process_data(path, ",", train_path, test_path)
                new_train_path="Baseline/res/{}_RBPR_train{}.csv".foramt(name,i)
                transdata(train_path,new_train_path)
                #model traing...
                model = RBPR(new_train_path, resultSaveFile, '\t', dim, iter)
                model.train_rbpr()
                model.name=Method[0]
                # 进行评价
                ndcgs,maps=ndcg_map_evulations_normal(model,Ks1,Ks2,train_path,test_path,n_items)
                
                for k in Ks1:
                    All_ndcgs[0][k].append(np.mean(ndcgs[k]))
                for k in Ks2:
                    All_maps[0][k].append(np.mean(maps[k]))

            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)
    

def evulation_mlls():
    
    print("start to evulation_mlls_datasets RBPR...")
    Dims=[10,20,30,40,50,60,80,100]
    experiment="Baseline/res/mlls.exps"
    resultSaveFile="Baseline/res/tmp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    Method=["RBRP"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")       
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            if name not in ["mlls"]:
                continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  

            for i in range(5):
                process_data_by_split(path,",",n_items,train_path,test_path,n_ng=4)
                
                process_data(path, ",", train_path, test_path)
                new_train_path="Baseline/res/{}_RBPR_train{}.csv".format(name,i)
                transdata(train_path,new_train_path)
                #model traing...
                model = RBPR(new_train_path, resultSaveFile, '\t', dim, iter)
                model.train_rbpr()
                model.name=Method[0]
                # 进行评价
                ndcgs=ndcg_evaluations(model,Ks1,test_path)
                
                for k in Ks1:
                    All_ndcgs[0][k].append(np.mean(ndcgs[k]))


            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)
    

def evulation_all_datasets_time():
    
    print("start to evulation_all_datasets RBPR time...")
    Dims=[10,20,30,40,50,60,80,100]
    experiment="Baseline/res/exps_time"
    resultSaveFile="Baseline/res/tmp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    Method=["RBRP"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")       
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            if name not in ["ml10m"]:
                continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            for i in range(1):
                # process_data_by_split(path,",",n_items,train_path,test_path,n_ng=4)
                process_data(path, ",", train_path, test_path)
                new_train_path="Baseline/res/RBPR_train.csv"
                transdata(train_path,new_train_path)
                #model traing...
                model = RBPR(new_train_path, resultSaveFile, '\t', dim, iter)
                model.train_rbpr()
                model.name=Method[0]
                # 进行评价
                # ndcgs,maps=ndcg_map_evulations_normal(model,Ks1,Ks2,train_path,test_path,n_items)
                ndcgs=ndcg_evaluations(model,Ks1,test_path)


                for k in Ks1:
                    All_ndcgs[0][k].append(np.mean(ndcgs[k]))

            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)


def evulation_all_datasets_save(rebuildDataset=False):
    
    print("start to evulation_all_datasets RBPR to save...")
    experiment="Baseline/RBRP/res/exps_save"
    resultSaveFile="Baseline/RBRP/res/tmp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    Method=["RBRP"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")       
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
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
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  

            for i  in range(1):
                train_path = train_path_format.format(i)
                test_path = test_path_format.format(i)
                # new_train_path="Baseline/res/RBPR_train.csv"
                # transdata(train_path,new_train_path)
                #model traing...
                model = RBPR(train_path, resultSaveFile, ',', dim, iter)
                model.train_rbpr()
                model.name=Method[0]
                # 进行评价
                # ndcgs,maps=ndcg_map_evulations_normal(model,Ks1,Ks2,train_path,test_path,n_items)
                ndcgs=ndcg_evaluations(model,Ks1,test_path)

                for k in Ks1:
                    All_ndcgs[0][k].append(np.mean(ndcgs[k]))
                save_path="res_save/{}_{}_{}.csv".format(model.name,name,i)
                pred_to_save(model,test_path,save_path)
            
            for method in ms: 
                print("method {}ndcg{}:{},ndcg{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10])
                ),file=f,flush=True)


if __name__ == "__main__":
    evulation_all_datasets_save()
    # evulation_mlls()
    # evulation_all_datasets()
    # evulation_all_datasets_dims()
    # evulation_all_datasets_time()
    pass
