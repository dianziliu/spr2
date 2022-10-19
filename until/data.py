"""
    用户处理划分数据集，要求数据集采用csv格式存储，
    标题字段有：['userId','movieId','rating',...]
    train_file:[u:用户,i:正样本,j:相似样本,k:负样本]
    test_file:['userId','movieId','rating',...]
    ng:dict[userId:[k:负样本]]
"""

from random import choices,choice,randint

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

"ratings.csv","::"
"PR\\train_file.csv"
"PR\\test_file.csv"
"PR\\test_ng.dict"


def process_data(path, sep, train_path, test_path, test_size=0.5, new_sep=",", index=False, header=True,):

    df = pd.read_csv(path, sep=sep)

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for uid, group in df.groupby(["userId"]):

        train, test = train_test_split(group, test_size=test_size)

        df1 = pd.concat([df1, train])
        df2 = pd.concat([df2, test])

    df1.to_csv(train_path, index=index)

    # 测试集保持不变
    df2.to_csv(test_path, index=index)



def process_data_by_numN(path,sep,n_items,N,train_path,test_path,ngs_path,n_ng=4):

    df=pd.read_csv(path,sep=sep)

    train_file=open(train_path,"w")
    test_file=open(test_path,"w")
    test_ng=ngs_path
    d=dict()

    df2=pd.DataFrame()

    iids=set([i for i in range(n_items)])

    for uid,group in df.groupby(["userId"]):
        h=list(group.movieId.unique())
        l=len(h)
        if l<=N:
            continue
        # 采集负样本范围
        ng=list(set(iids)-set(h))
        ngs=choices(ng,k=l*5)
        # 划分训练集测试集
        g=[i for i in range(l)]
        test_index=choices(g,k=N)
        test=group.ix[test_index]
        for one in test_index:
            del h[one]
        train=h
        # 写入训练集
        pgs=choices(train,k=l*5)
        for i in range(len(train)):
            for j in range(n_ng): #here control num of negative sample
                train_file.write("{},{},{},{}\n".format(
                    uid, train[i], pgs[i*n_ng+j], ngs[i*n_ng+j]))
        df2=pd.concat([df2,test]) 

    train_file.close()
    df2.to_csv(test_file,index=False)
    joblib.dump(d,test_ng)

  

def process_data_with_ngs(path,sep,n_items,train_path,test_path,ngs_path,test_size=0.2,n_ng=4):

    df=pd.read_csv(path,sep=sep)

    train_file=open(train_path,"w")
    test_file=open(test_path,"w")
    test_ng=ngs_path
    d=dict()

    df2=pd.DataFrame()
    iids=[i for i in range(n_items)]

    for uid,group in df.groupby(["userId"]):
        h=list(group.movieId.unique())
        l=len(h)
        ng=list(set(iids)-set(h))
        ngs=choices(ng,k=l*5)
        
        train,test=train_test_split(group,test_size=test_size)
        train=list(train.movieId.unique())
        
        pgs=choices(train,k=l*5)
        for i in range(len(train)):
            for j in range(n_ng): #here control num of negative sample
                train_file.write("{},{},{},{}\n".format(
                    uid, train[i], pgs[i*n_ng+j], ngs[i*n_ng+j]))
        
        ngs=choices(ng,k=100) #here control num of negative sample
        d[uid]=ngs
        df2=pd.concat([df2,test])
            

    train_file.close()
    df2.to_csv(test_file,index=False)
    joblib.dump(d,test_ng)

def process_data_by_split(path,sep,n_items,train_path,test_path,test_size=0.5,n_ng=4):
    
    """
    训练文件以四元组的形式保存 "u,i,postive,negative"
    测试集以原始文件格式保存.csv
    """
    df=pd.read_csv(path,sep=sep,engine="python")

    train_file=open(train_path,"w")
    # test_file=open("test_file.csv","w")
    
    df2=pd.DataFrame()

    iids=[i for i in range(n_items)]
    for uid,group in tqdm(df.groupby(["userId"]),ncols=100):
        
        h=list(group.movieId.unique())
        l=len(h)
        ng=list(set(iids)-set(h))
        ngs=choices(ng,k=l*5)
        
        train,test=train_test_split(group,test_size=test_size)
        train=list(train.movieId.unique())
        pgs=choices(train,k=l*5)
        for i in range(len(train)):
            for j in range(n_ng):
                train_file.write("{},{},{},{}\n".format(
                    uid, train[i], pgs[i*n_ng+j], ngs[i*n_ng+j]))
        df2=pd.concat([df2,test])

    train_file.close()
    df2.to_csv(test_path,index=False)



def epoch_process_data_by_split(epochs,path,sep,n_items,train_path,test_path,test_size=0.5,n_ng=4):
    
    """
    训练文件以四元组的形式保存 "u,i,postive,negative"
    测试集以原始文件格式保存.csv
    """
    df=pd.read_csv(path,sep=sep,engine="python")

    iids=[i for i in range(n_items)]
    
    for epoch in range(epochs):
        print("epoch {} processing...".format(epoch))
        train_file=open(train_path.format(epoch),"w")
        # test_file=open("test_file.csv","w")
        
        df2=pd.DataFrame()

        
        for uid,group in tqdm(df.groupby(["userId"]),ncols=50):
            
            h=list(group.movieId.unique())
            l=len(h)
            ng=list(set(iids)-set(h))
            ngs=choices(ng,k=l*5)
            
            train,test=train_test_split(group,test_size=test_size)
            train=list(train.movieId.unique())
            pgs=choices(train,k=l*5)
            for i in range(len(train)):
                for j in range(n_ng):
                    train_file.write("{},{},{},{}\n".format(
                        uid, train[i], pgs[i*n_ng+j], ngs[i*n_ng+j]))
            df2=pd.concat([df2,test])

        train_file.close()
        df2.to_csv(test_path.format(epoch),index=False)
