import os
from time import sleep
from datetime import datetime
import numpy as np

from model import Simi
from until.data import process_data_by_split,epoch_process_data_by_split
from until.evulation import (ndcg_evaluations, ndcg_map_evulations,pred_to_save,
                             pr_ndcg_evulation_Alls)

train_path = "run/train.csv"
test_path = "run/test.csv"
u_path = "run/u{}.bin"
i_path = "run/i{}.bin"

Method = ["SPR",
          "J-SPR"]


# paths = ["data/Ml100Krating.csv",
#          "data/ML1Mratings.csv",
#          "data/YahooR3.csv",
#          "data/YahooR4.csv"]
# n_userss=[6400,6400,16000,8000]
# n_itmess=[4000,4000,1024,12000]
# dataset=["ml100k","ml1m","r3","r4"]

paths = ["data/MLLSratings.csv",
         "data/ML10Mratings.csv",
         "data/Ml100Krating.csv",
         "data/ML1Mratings.csv",
         "data/YahooR3.csv",
         "data/YahooR4.csv"]

                      

# ML100K,ML1M,YahooR3,YahooR4
n_userss=[800,80000,6400,6400,16000,8000]
n_itmess=[9600,80000,4000,4000,1024,12000]

dataset=["mlls","ml10m","ml100k","ml1m","r3","r4"]

dim=10
lr=0.007
rg=0.05
iter=20

def test_a_b():
    log_file="hp_a_b.log"
    experiment="hp_a_b_exps"
    hys=[(0.2,0.2),(0.3,0.4),(0.4,0.4),(0.5,0.5)]

    Ks1=[5,10]
    Ks2=[10,20]
    exe=r"E:\设计文件\joint\MF2.exe"
    method=3
    with open(experiment,"a") as f:
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            All_ndcgs = [{5: [], 10: []} for i in range(3)]
            All_maps = [{10: [], 20: []} for i in range(3)]
            print("dataset",name)
            # print("dataset",name,file=f)
            for i in range(1):
                # process_data_by_split(path,",",n_items,train_path,test_path)
                this_method=Method[method]
                upath=u_path.format(method)
                ipath=i_path.format(method)
                for index,ab in enumerate(hys):
                    a,b=ab
                    cmd="{} -Method {} -debug 1 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    -train_path {} -test_path {} -u_path {} -i_path {} -log_path {} -a {} -b {}".format(
                        exe,method,n_users,n_items,dim,iter,lr,rg,
                        train_path,test_path,upath,ipath,log_file,a,b
                    )
                    os.system(cmd)
                    sleep(10)
                    model=Simi(method,n_users,n_items,dim,train_path)
                    model.load(upath,ipath)
                    model.name=this_method
                    ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                    for k in Ks1:
                        All_ndcgs[index][k].append(np.mean(ndcgs[k]))
                    for k in Ks2:
                        All_maps[index][k].append(np.mean(maps[k]))
            for index,ab in enumerate(hys):
                a,b=ab
                print("method {},a={},b={}\n,ndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],a,b,5,np.mean(All_ndcgs[index][5]),10,np.mean(All_ndcgs[index][10]),
                    10,np.mean(All_maps[index][10]),20,np.mean(All_maps[index][20])
                ))

            break

def evulation_all_datasets():    

    experiment="run/both_exps"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0,1]
    exes=["c_model/SPR","c_model/jSPR"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            # if name not in ["r3"]:
            #     continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            for i in range(5):
                process_data_by_split(path,",",n_items,train_path,test_path)
                for m in ms:
                    method_name=Method[m]
                    upath=u_path.format(m)
                    ipath=i_path.format(m)
                    exe=exes[m]
                    # cmd="{} -Method {} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    # -train_path {} -test_path {} -u_path {} -i_path {} -log_path {} -a 0.1 -b 0.1".format(
                    #     exe,m,n_users,n_items,dim,iter,lr,rg,
                    #     train_path,test_path,upath,ipath,log_file
                    # )
                    cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    -train_path {} -test_path {} -u_path {} -i_path {} -a 0.09 -b 0.83".format(
                        exe,n_users,n_items,dim,iter,lr,rg,
                        train_path,test_path,upath,ipath
                    )
                    os.system(cmd)
                    model=Simi(m,n_users,n_items,dim,train_path)
                    model.load(upath,ipath)
                    model.name=method_name
                    ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                    for k in Ks1:
                        All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                    for k in Ks2:
                        All_maps[m][k].append(np.mean(maps[k]))

            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)
 
def evulation_all_datasets_SPR():    

    experiment="run/hpy_exp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[1]
    exes=["c_model/SPR"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            if name not in ["ml100k"]:
                continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            for i in range(5):
                process_data_by_split(path,",",n_items,train_path,test_path)
                range_a=[1.0*i/100 for i in range(1,30)]
                f.write('a:'+str(a)+'\n')
                for a in range_a:
                    for m in ms:
                        method_name=Method[m]
                        upath=u_path.format(m)
                        ipath=i_path.format(m)
                        exe=exes[m]
        
                        cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        -train_path {} -test_path {} -u_path {} -i_path {} -a {}".format(
                            exe,n_users,n_items,dim,iter,lr,rg,
                            train_path,test_path,upath,ipath,a
                        )
                        os.system(cmd)
                        model=Simi(m,n_users,n_items,dim,train_path)
                        model.load(upath,ipath)
                        model.name=method_name
                        ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                        for k in Ks1:
                            All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                        for k in Ks2:
                            All_maps[m][k].append(np.mean(maps[k]))
                    for method in ms: 
                        print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                            Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                            10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                        ),file=f,flush=True)

def evulation_all_datasets_jSPR():    

    experiment="run/jspr_exps"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[1]
    exes=["c_model/SPR","c_model/jSPR"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            for i in range(5):
                process_data_by_split(path,",",n_items,train_path,test_path)
                for m in ms:
                    method_name=Method[m]
                    upath=u_path.format(m)
                    ipath=i_path.format(m)
                    exe=exes[m]
                    # cmd="{} -Method {} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    # -train_path {} -test_path {} -u_path {} -i_path {} -log_path {} -a 0.1 -b 0.1".format(
                    #     exe,m,n_users,n_items,dim,iter,lr,rg,
                    #     train_path,test_path,upath,ipath,log_file
                    # )
                    cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    -train_path {} -test_path {} -u_path {} -i_path {} -a 0.1 -b 0.065".format(
                        exe,n_users,n_items,dim,iter,lr,rg,
                        train_path,test_path,upath,ipath
                    )
                    os.system(cmd)
                    model=Simi(m,n_users,n_items,dim,train_path)
                    model.load(upath,ipath)
                    model.name=method_name
                    ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                    for k in Ks1:
                        All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                    for k in Ks2:
                        All_maps[m][k].append(np.mean(maps[k]))
            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)


def evulation_all_datasets_ngs():
    

    train_path = "run/ng_train.csv"
    test_path = "run/ng_test.csv"
    u_path = "run/ng_u{}.bin"
    i_path = "run/ng_i{}.bin"

    ngs=[1,2,4,6,8,10]
    # ngs=[4]
    experiment="run/ng_both_exps"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0,1]
    exes=["c_model/SPR","c_model/jSPR"]

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for ng in ngs:
            f.write("ng:"+str(ng)+"\n")
            for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
                # if name not in ["r3"]:
                #     continue
                # traversing each datset
                # evulation for each dataset
                All_ndcgs = [{5: [], 10: []} for i in range(8)]
                All_maps = [{10: [], 20: []} for i in range(8)]
                print("dataset",name)
                print("dataset",name,file=f,flush=True)  
                for i in range(5):
                    process_data_by_split(path,",",n_items,train_path,test_path,n_ng=ng)
                    for m in ms:
                        method_name=Method[m]
                        upath=u_path.format(m)
                        ipath=i_path.format(m)
                        exe=exes[m]

                        cmd = "{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        -train_path {} -test_path {} -u_path {} -i_path {} -a 0.21 -b 0.54".format(
                            exe, n_users, n_items, dim, iter, lr, rg,
                            train_path, test_path, upath, ipath
                        )
                        os.system(cmd)
                        model=Simi(m,n_users,n_items,dim,train_path)
                        model.load(upath,ipath)
                        model.name=method_name
                        ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                        for k in Ks1:
                            All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                        for k in Ks2:
                            All_maps[m][k].append(np.mean(maps[k]))

                for method in ms: 
                    print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                        Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                        10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                    ),file=f,flush=True)
    
    
def evulation_all_datasets_a():
    
    experiment="run/hpy_exp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    exes=["c_model/SPR"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            if name not in ["ml100k"]:
                continue
            # traversing each datset
            # evulation for each dataset

            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            range_a=[1.0*i/100 for i in range(1,31)]
            for a in range_a:
                f.write('a:'+str(a)+'\n')
                All_ndcgs = [{5: [], 10: []} for i in range(8)]
                All_maps = [{10: [], 20: []} for i in range(8)]
                for i in range(1):
                    process_data_by_split(path,",",n_items,train_path,test_path)
                    for m in ms:
                        method_name=Method[m]
                        upath=u_path.format(m)
                        ipath=i_path.format(m)
                        exe=exes[m]
                        cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        -train_path {} -test_path {} -u_path {} -i_path {} -a {}".format(
                            exe,n_users,n_items,dim,iter,lr,rg,
                            train_path,test_path,upath,ipath,a
                        )
                        os.system(cmd)
                        model=Simi(m,n_users,n_items,dim,train_path)
                        model.load(upath,ipath)
                        model.name=method_name
                        ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                        for k in Ks1:
                            All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                        for k in Ks2:
                            All_maps[m][k].append(np.mean(maps[k]))
                    for method in ms: 
                        print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                            Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                            10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                        ),file=f,flush=True)   

def evulation_all_datasets_b():
     
    experiment="run/b_hpy_exp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    exes=["c_model/jSPR"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            if name not in ["ml100k"]:
                continue
            # traversing each datset
            # evulation for each dataset

            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            range_b=[1.0*i/100 for i in range(1,31)]
            for b in range_b:
                b=1-0.21-b
                f.write('b:'+str(b)+'\n')
                All_ndcgs = [{5: [], 10: []} for i in range(8)]
                All_maps = [{10: [], 20: []} for i in range(8)]
                for i in range(1):
                    process_data_by_split(path,",",n_items,train_path,test_path)
                    for m in ms:
                        method_name=Method[m]
                        upath=u_path.format(m)
                        ipath=i_path.format(m)
                        exe=exes[m]
                        cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        -train_path {} -test_path {} -u_path {} -i_path {} -a 0.21 -b {}".format(
                            exe,n_users,n_items,dim,iter,lr,rg,
                            train_path,test_path,upath,ipath,b
                        )
                        os.system(cmd)
                        model=Simi(m,n_users,n_items,dim,train_path)
                        model.load(upath,ipath)
                        model.name=method_name
                        ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                        for k in Ks1:
                            All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                        for k in Ks2:
                            All_maps[m][k].append(np.mean(maps[k]))
                    for method in ms: 
                        print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                            Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                            10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                        ),file=f,flush=True)   


def evulation_all_datasets_dim():
    
    train_path = "run/dim_train.csv"
    test_path = "run/dim_test.csv"
    u_path = "run/dim_u{}.bin"
    i_path = "run/dim_i{}.bin"

    print("start to evulation_all_datasets_dim")
    Dims=[10,20,30,40,50,60,80,100]
    experiment="run/dim_both_exps"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0,1]
    exes=["c_model/SPR","c_model/jSPR"]

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for dim in Dims:
            f.write("dim:"+str(dim)+"\n")
            for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
                # if name not in ["r3"]:
                #     continue
                # traversing each datset
                # evulation for each dataset
                All_ndcgs = [{5: [], 10: []} for i in range(8)]
                All_maps = [{10: [], 20: []} for i in range(8)]
                print("dataset",name)
                print("dataset",name,file=f,flush=True)  
                for i in range(5):
                    process_data_by_split(path,",",n_items,train_path,test_path,n_ng=2)
                    for m in ms:
                        method_name=Method[m]
                        upath=u_path.format(m)
                        ipath=i_path.format(m)
                        exe=exes[m]

                        cmd = "{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        -train_path {} -test_path {} -u_path {} -i_path {} -a 0.09 -b 0.83".format(
                            exe, n_users, n_items, dim, iter, lr, rg,
                            train_path, test_path, upath, ipath
                        )
                        os.system(cmd)
                        model=Simi(m,n_users,n_items,dim,train_path)
                        model.load(upath,ipath)
                        model.name=method_name
                        ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                        for k in Ks1:
                            All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                        for k in Ks2:
                            All_maps[m][k].append(np.mean(maps[k]))

                for method in ms: 
                    print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                        Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                        10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                    ),file=f,flush=True)


def save_all_res():
    print("start to run save_all_res!")
    train_path = "run/dim_train.csv"
    test_path  = "run/dim_test.csv"
    u_path     = "run/dim_u{}.bin"
    i_path     = "run/dim_i{}.bin"

    print("start to evulation_all_datasets_and save")
    Dims       = [10]
    experiment = "res_save/both_exps"
    Ks1        = [5,10]
    Ks2        = [10,20]
    ms         = [0,1]
    exes       = ["c_model/SPR","c_model/jSPR"]

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for dim in Dims:
            f.write("dim:"+str(dim)+"\n")
            for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):

                All_ndcgs = [{5: [], 10: []} for i in range(8)]
                All_maps = [{10: [], 20: []} for i in range(8)]
                print("dataset",name)
                print("dataset",name,file=f,flush=True)  
                for i in range(5):
                    process_data_by_split(path,",",n_items,train_path,test_path,n_ng=2)
                    for m in ms:
                        method_name=Method[m]
                        upath=u_path.format(m)
                        ipath=i_path.format(m)
                        exe=exes[m]

                        cmd = "{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        -train_path {} -test_path {} -u_path {} -i_path {} -a 0.09 -b 0.83".format(
                            exe, n_users, n_items, dim, iter, lr, rg,
                            train_path, test_path, upath, ipath
                        )
                        os.system(cmd)
                        model=Simi(m,n_users,n_items,dim,train_path)
                        model.load(upath,ipath)
                        model.name=method_name
                        ndcgs,maps=ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_items)
                        for k in Ks1:
                            All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                        for k in Ks2:
                            All_maps[m][k].append(np.mean(maps[k]))
                        # 
                        save_path="res_save/{}_{}_{}.csv".format(method_name,name,i)
                        pred_to_save(model,test_path,save_path)

                for method in ms: 
                    print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                        Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                        10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                    ),file=f,flush=True)



def evulation_ml10m():    

    print("Start to run ml10m")
    experiment="run/ml10m.exp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0,1]
    exes=["c_model/SPR","c_model/jSPR"]

    # split dataset
    # print("Start to split dataset")
    # for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
    #     # train_path = "run/m10/train{}.csv"
    #     # test_path = "run/m10/test{}.csv"
    #     dir=os.path.join("run",name)
    #     if not os.path.exists(dir):
    #         os.mkdir(dir)
    #     train_path=os.path.join("run",name,"train{}.csv")
    #     test_path=os.path.join("run",name,"test{}.csv")
    #     epoch_process_data_by_split(5,path,",",n_items,train_path,test_path)
        

    # run 
    print("Start to training...")
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            # if name not in ["r3"]:
            #     continue

            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  

            train_path=os.path.join("run",name,"train{}.csv")
            test_path=os.path.join("run",name,"test{}.csv")

            for i in range(5):
                # train_path = "run/m10/train{}.csv".format(i)
                # test_path = "run/m10/test{}.csv".format(i)
                # process_data_by_split(path,",",n_items,train_path,test_path)
                for m in ms:
                    method_name=Method[m]
                    upath=u_path.format(m)
                    ipath=i_path.format(m)
                    exe=exes[m]
                    # cmd="{} -Method {} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    # -train_path {} -test_path {} -u_path {} -i_path {} -log_path {} -a 0.1 -b 0.1".format(
                    #     exe,m,n_users,n_items,dim,iter,lr,rg,
                    #     train_path,test_path,upath,ipath,log_file
                    # )
                    cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    -train_path {} -test_path {} -u_path {} -i_path {} -a 0.09 -b 0.83".format(
                        exe,n_users,n_items,dim,iter,lr,rg,
                        train_path.format(i),test_path.format(i),upath,ipath
                    )
                    os.system(cmd)
                    model=Simi(m,n_users,n_items,dim,train_path)
                    model.load(upath,ipath)
                    model.name=method_name
                    ndcgs, maps = ndcg_map_evulations(
                        model, Ks1, Ks2, train_path.format(i), test_path.format(i), n_items)
                    for k in Ks1:
                        All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                    for k in Ks2:
                        All_maps[m][k].append(np.mean(maps[k]))
            
            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)
 
def evulation_ml10m_ndcg_only():    

    """
    只计算了ndcg的评价指标
    """
    print("Start to run ml10m")
    experiment="run/ml10m_ndcg.exp.time"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0,1]
    exes=["c_model/SPR","c_model/jSPR"]

    # split dataset
    # print("Start to split dataset")
    # for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
    #     # train_path = "run/m10/train{}.csv"
    #     # test_path = "run/m10/test{}.csv"
    #     dir=os.path.join("run",name)
    #     if not os.path.exists(dir):
    #         os.mkdir(dir)
    #     train_path=os.path.join("run",name,"train{}.csv")
    #     test_path=os.path.join("run",name,"test{}.csv")
    #     epoch_process_data_by_split(5,path,",",n_items,train_path,test_path)
        

    # run 
    print("Start to training m10ndcg...")
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            # if name not in ["r3"]:
            #     continue

            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  

            train_path=os.path.join("run",name,"train{}.csv")
            test_path=os.path.join("run",name,"test{}.csv")

            for i in range(1):
                # train_path = "run/m10/train{}.csv".format(i)
                # test_path = "run/m10/test{}.csv".format(i)
                # process_data_by_split(path,",",n_items,train_path,test_path)
                for m in ms:
                    method_name=Method[m]
                    upath=u_path.format(m)
                    ipath=i_path.format(m)
                    exe=exes[m]
                    # cmd="{} -Method {} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    # -train_path {} -test_path {} -u_path {} -i_path {} -log_path {} -a 0.1 -b 0.1".format(
                    #     exe,m,n_users,n_items,dim,iter,lr,rg,
                    #     train_path,test_path,upath,ipath,log_file
                    # )
                    cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    -train_path {} -test_path {} -u_path {} -i_path {} -a 0.09 -b 0.83".format(
                        exe,n_users,n_items,dim,iter,lr,rg,
                        train_path.format(i),test_path.format(i),upath,ipath
                    )
                    os.system(cmd)
                    model=Simi(m,n_users,n_items,dim,train_path)
                    model.load(upath,ipath)
                    model.name=method_name
                    ndcgs=ndcg_evaluations(model,Ks1,test_path.format(i))
                    # ndcgs, maps = ndcg_map_evulations(
                    #    model, Ks1, Ks2, train_path.format(i), test_path.format(i), n_items)
                    for k in Ks1:
                        All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                    # for k in Ks2:
                    #     All_maps[m][k].append(np.mean(maps[k]))
            
            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)
            break
 
 
def evulation_all_datasets_time():    

    print("start to run time evulation...")
    experiment="run/time_exp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0,1]
    exes=["c_model/SPR","c_model/jSPR"]
    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
            # if name not in ["r3"]:
            #     continue
            # traversing each datset
            # evulation for each dataset
            All_ndcgs = [{5: [], 10: []} for i in range(8)]
            All_maps = [{10: [], 20: []} for i in range(8)]
            print("dataset",name)
            print("dataset",name,file=f,flush=True)  
            for i in range(1):
                process_data_by_split(path,",",n_items,train_path,test_path)
                for m in ms:
                    method_name=Method[m]
                    upath=u_path.format(m)
                    ipath=i_path.format(m)
                    exe=exes[m]
                    # cmd="{} -Method {} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    # -train_path {} -test_path {} -u_path {} -i_path {} -log_path {} -a 0.1 -b 0.1".format(
                    #     exe,m,n_users,n_items,dim,iter,lr,rg,
                    #     train_path,test_path,upath,ipath,log_file
                    # )
                    cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                    -train_path {} -test_path {} -u_path {} -i_path {} -a 0.09 -b 0.83".format(
                        exe,n_users,n_items,dim,iter,lr,rg,
                        train_path,test_path,upath,ipath
                    )
                    os.system(cmd)
                    model=Simi(m,n_users,n_items,dim,train_path)
                    model.load(upath,ipath)
                    model.name=method_name
                    # 只计算NDCG
                    ndcgs=ndcg_evaluations(model,Ks1,test_path)
                    for k in Ks1:
                        All_ndcgs[m][k].append(np.mean(ndcgs[k]))


            for method in ms: 
                print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                    Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                    10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                ),file=f,flush=True)
 


def evulation_ml10m_dim():
    train_path = "run/dim_train.csv"
    test_path = "run/dim_test.csv"
    u_path = "run/m10_dim_u{}.bin"
    i_path = "run/m10_dim_i{}.bin"

    print("start to evulation_ml10m_dim")
    Dims=[10,20,30,40,50,60,80,100]
    experiment="run/ml_10m_dim.exp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0]
    exes=["c_model/SPR","c_model/jSPR"]

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for dim in Dims:
            f.write("dim:"+str(dim)+"\n")
            for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
                if name not in ["ml10m"]:
                    continue
                # traversing each datset
                # evulation for each dataset
                All_ndcgs = [{5: [], 10: []} for i in range(8)]
                All_maps = [{10: [], 20: []} for i in range(8)]
                print("dataset",name)
                print("dataset",name,file=f,flush=True)  

                train_path=os.path.join("run",name,"train{}.csv")
                test_path=os.path.join("run",name,"test{}.csv")

                for i in range(5):
                    # train_path = "run/m10/train{}.csv".format(i)
                    # test_path = "run/m10/test{}.csv".format(i)
                    # process_data_by_split(path,",",n_items,train_path,test_path)
                    for m in ms:
                        method_name=Method[m]
                        upath=u_path.format(m)
                        ipath=i_path.format(m)
                        exe=exes[m]
                        # cmd="{} -Method {} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        # -train_path {} -test_path {} -u_path {} -i_path {} -log_path {} -a 0.1 -b 0.1".format(
                        #     exe,m,n_users,n_items,dim,iter,lr,rg,
                        #     train_path,test_path,upath,ipath,log_file
                        # )
                        cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        -train_path {} -test_path {} -u_path {} -i_path {} -a 0.09 -b 0.83".format(
                            exe,n_users,n_items,dim,iter,lr,rg,
                            train_path.format(i),test_path.format(i),upath,ipath
                        )
                        os.system(cmd)
                        model=Simi(m,n_users,n_items,dim,train_path)
                        model.load(upath,ipath)
                        model.name=method_name
                        ndcgs=ndcg_evaluations(model,Ks1,test_path.format(i))
                        # ndcgs, maps = ndcg_map_evulations(
                        #    model, Ks1, Ks2, train_path.format(i), test_path.format(i), n_items)
                        for k in Ks1:
                            All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                        # for k in Ks2:
                        #     All_maps[m][k].append(np.mean(maps[k]))
            


                for method in ms: 
                    print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                        Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                        10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                    ),file=f,flush=True)



def evulation_mlls_dim():
    train_path = "run/ls_dim_train.csv"
    test_path = "run/ls_dim_test.csv"
    # u_path = "run/m1ls_dim_u{}.bin"
    # i_path = "run/m1ls_dim_i{}.bin"

    print("start to evulation_ml10m_dim")
    Dims=[10,20,30,40,50,60,80,100]
    experiment="run/mlls_dim.exp"
    Ks1=[5,10]
    Ks2=[10,20]
    ms=[0,1]
    exes=["c_model/SPR","c_model/jSPR"]

    with open(experiment,"a") as f:
        f.write(str(datetime.now())+"\n")
        for dim in Dims:
            f.write("dim:"+str(dim)+"\n")
            for name,path,n_users,n_items in zip(dataset,paths,n_userss,n_itmess):
                if name not in ["mlls"]:
                    continue
                # traversing each datset
                # evulation for each dataset
                All_ndcgs = [{5: [], 10: []} for i in range(8)]
                All_maps = [{10: [], 20: []} for i in range(8)]
                print("dataset",name)
                print("dataset",name,file=f,flush=True)  

                # train_path=os.path.join("run",name,"train{}.csv")
                # test_path=os.path.join("run",name,"test{}.csv")

                for i in range(5):
                    train_path = "run/mlls/train{}.csv".format(i)
                    test_path = "run/mlls/test{}.csv".format(i)
                    process_data_by_split(path,",",n_items,train_path,test_path)
                    for m in ms:
                        method_name=Method[m]
                        upath=u_path.format(m)
                        ipath=i_path.format(m)
                        exe=exes[m]
                        # cmd="{} -Method {} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        # -train_path {} -test_path {} -u_path {} -i_path {} -log_path {} -a 0.1 -b 0.1".format(
                        #     exe,m,n_users,n_items,dim,iter,lr,rg,
                        #     train_path,test_path,upath,ipath,log_file
                        # )
                        cmd="{} -debug 0 -n_users {} -n_items {} -dim {} -iter {} -lr {} -rg {} \
                        -train_path {} -test_path {} -u_path {} -i_path {} -a 0.09 -b 0.83".format(
                            exe,n_users,n_items,dim,iter,lr,rg,
                            train_path,test_path,upath,ipath
                        )
                        os.system(cmd)
                        model=Simi(m,n_users,n_items,dim,train_path)
                        model.load(upath,ipath)
                        model.name=method_name
                        ndcgs=ndcg_evaluations(model,Ks1,test_path)
                        # ndcgs, maps = ndcg_map_evulations(
                        #    model, Ks1, Ks2, train_path.format(i), test_path.format(i), n_items)
                        for k in Ks1:
                            All_ndcgs[m][k].append(np.mean(ndcgs[k]))
                        # for k in Ks2:
                        #     All_maps[m][k].append(np.mean(maps[k]))
            


                for method in ms: 
                    print("method {},\nndcg{}:{},ndcg{}:{}\nmap{}:{},map{}:{}".format(
                        Method[method],5,np.mean(All_ndcgs[method][5]),10,np.mean(All_ndcgs[method][10]),
                        10,np.mean(All_maps[method][10]),20,np.mean(All_maps[method][20])
                    ),file=f,flush=True)


def evulation_one_data():
    pass


if __name__ == "__main__":
    evulation_mlls_dim()
    # evulation_ml10m_dim()
    # evulation_ml10m_ndcg_only()
    # evulation_all_datasets_time()
    # evulation_ml10m()

    # save_all_res()
    # evulation_all_datasets_dim()
    # evulation_all_datasets_ngs()
    # evulation_all_datasets()
    # evulation_all_datasets_SPR()
    # evulation_all_datasets_jSPR()

    # evulation_all_datasets_a()
    # evulation_all_datasets_b()
