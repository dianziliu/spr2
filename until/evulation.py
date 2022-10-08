from math import log2
from random import shuffle
import numpy as np
import pandas as pd
import joblib
from multiprocessing.pool import Pool
from tqdm import tqdm,tqdm_pandas

"""
需要注意到，NDCG,MAP存在多种计算方式
"""


def AP(K, scores, ids, method=0):
    """
    scores:[iId,score,rating]
    ids:用户的测试集
    """
    s=sorted(scores,key=lambda x: x[1],reverse=True)
    s=[i[0] for i in s[:K]] 
    sum=0
    hits=0
    for i in range(K):
        if s[i] in ids:
            hits+=1
            sum+=1.0*hits/(i+1)
    if hits==0:
        return 0
    else:
        if method:
            K=K if K<len(ids) else len(ids) # add this check, version by LibRec
            return sum/K
        return sum/hits    

def MAP_evulation_Alls(model, Ks:list, test_path,n_itmes):
    aps=dict()
    for K in Ks:
        aps[K]=[]
    df = pd.read_csv(test_path)

    for uid, group in df.groupby(["userId"]):
        scores = []
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # ids 为用户历史行为
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
        shuffle(scores)
        for K in Ks:
            aps[K].append(AP(K,scores,ids))
    for K in Ks:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))


def DCG(K, l, w,method=0):
    # [[iId,score,rating],...]
    s = sorted(l, key=lambda x: x[1], reverse=True)
    if method:
        x=[i[2] for i in s[:K]]
    else:
        x=[pow(2,i[2])-1 for i in s[:K]] #fixed this bug
    return np.dot(x, w[:K])

def iDCG(K, l, w,method=0):
    s = sorted(l, key=lambda x: x[2], reverse=True)
    if method:
        x=[i[2] for i in s[:K]]
    else:
        x=[pow(2,i[2])-1 for i in s[:K]]
    return np.dot(x, w)

def nDCG(K, l,method=0):
    """
    k:int
    l:list[(int ,int ,int)]
    id,pred_score,rating
    """
    if len(l)<K:
        K=len(l)
    w = [1/log2(2+i) for i in range(K)]
    dcg = DCG(K, l, w,method)
    idcg = iDCG(K, l, w,method)
    return dcg/idcg


def Precision_Recall(K,scores,right):
    s=sorted(scores,key=lambda x: x[1],reverse=True)
    s=[i[0] for i in s[:K]]
    m=set(s) & set(right)
    p=len(m)/K
    r=len(m)/len(right)
    return p,r


def pr_evulation(model, K, test_path, test_ng):
    precisions = []
    recalls = []
    df = pd.read_csv(test_path)
    ngs = joblib.load(test_ng)
    for uid, group in df.groupby(["userId"]):
        scores = []

        ids = group.movieId.unique()

        for i in ids:
            scores.append((i, model.predict(uid, i)))

        ng = ngs[uid]
        for j in ng:
            scores.append((i, model.predict(uid, j), 0))
        shuffle(scores)
        p, r = Precision_Recall(K, scores, ids)
        precisions.append(p)
        recalls.append(r)

    print("model{} Precision@{}:{},Recall@{}:{}".format(
        model.name, K, np.mean(precisions), K, np.mean(recalls)))


def ndcg_evaluation(model, K, test_file,method=0):
    df = pd.read_csv(test_file)
    ndcgs = []
    for uid, group in df.groupby(["userId"]):
        scores = []
        for row,one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores.append((i, model.predict(u, i), r))
        ndcgs.append(nDCG(K, scores,method))
    
    print("model{} nDCG@{}:{}".format(model.name, K, np.mean(ndcgs)))

def ndcg_evaluations(model, Ks, test_file,method=1):
    df = pd.read_csv(test_file)
    ndcgs=dict()
    for K in Ks:
        ndcgs[K]=[]

    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        scores = []
        for row,one in group.iterrows():
            u, i, r = [int(one["userId"]), int(one["movieId"]), one["rating"]]
            scores.append((i, model.predict(u, i), r))
        shuffle(scores)
        for K in Ks:
            ndcgs[K].append(nDCG(K,scores,method))
    for K in Ks:
        print("model{} nDCG@{}:{}".format(model.name, K, np.mean(ndcgs[K])))
    return ndcgs


def ndcg_evulations_pred(model_name,Ks,pred_path,method=0):
    df=pd.read_csv(pred_path)
    ndcgs=dict()
    for K in Ks:
        ndcgs[K]=[]

    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        scores = []
        for row,one in group.iterrows():
            u, i, r,pred = [int(one["userId"]), int(one["movieId"]), one["rating"],one["pred"]]
            scores.append((i, pred, r))
        shuffle(scores)
        for K in Ks:
            a=nDCG(K,scores,method)
            ndcgs[K].append(a)
    for K in Ks:
        print("model{} nDCG@{}:{}".format(model_name, K, np.mean(ndcgs[K])))
    return ndcgs   



def pr_ndcg_evulation(model, K, test_path, test_ng):
    precisions = []
    recalls = []
    ndcgs = []
    df = pd.read_csv(test_path)
    ngs = joblib.load(test_ng)
    for uid, group in df.groupby(["userId"]):
        scores = []
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores.append((i, model.predict(u, i), r))
        ng = ngs[uid]
        for j in ng:
            scores.append((i, model.predict(uid, j), 0))
        shuffle(scores)
        p, r = Precision_Recall(K, scores, ids)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(nDCG(K, scores))
    print("model{0} Precision@{1}:{2},Recall@{1}:{3},nDCG@{1}:{4}".format(
        model.name, K, np.mean(precisions), np.mean(recalls), np.mean(ndcgs)))


def pr_ndcg_evulation_All(model, K, test_path, n_itmes):
    precisions = []
    recalls = []
    ndcgs = []
    ndcgs2=[]
    df = pd.read_csv(test_path)
    # All_items=[i for i in range(n_itmes)]

    for uid, group in df.groupby(["userId"]):
        scores = []
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        p, r = Precision_Recall(K, scores, ids)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(nDCG(K, scores))
        ndcgs2.append(nDCG(K, scores2))
    print("model{0} Precision@{1}:{2},Recall@{1}:{3},nDCG@{1}:{4},nDCG2@{1}:{5}".format(
        model.name, K, np.mean(precisions), np.mean(recalls), np.mean(ndcgs),np.mean(ndcgs2)))

def pr_ndcg_evulation_Alls(model, Ks, test_path, n_itmes):
    precisions = {}
    recalls = {}
    ndcgs = {}
    ndcgs2={}
    for one in Ks:
        precisions[one] = []
        recalls[one] = []
        ndcgs[one] = []
        ndcgs2[one]=[]
    df = pd.read_csv(test_path)
    # All_items=[i for i in range(n_itmes)]

    for uid, group in df.groupby(["userId"]):
        scores = []
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        ids = group.movieId.unique()
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks:
            p, r = Precision_Recall(K, scores, ids)
            precisions[K].append(p)
            recalls[K].append(r)
            ndcgs[K].append(nDCG(K, scores))
            ndcgs2[K].append(nDCG(K, scores2))
    for K in Ks:
        print("model{0} Precision@{1}:{2},Recall@{1}:{3},nDCG@{1}:{4},nDCG2@{1}:{5}".format(
            model.name, K, np.mean(precisions[K]), np.mean(recalls[K]), np.mean(ndcgs[K]),np.mean(ndcgs2[K])))

def ndcg_map_evulations(model,Ks1,Ks2,train_path,test_path,n_itmes,method=1):

    aps=dict()
    ndcgs=dict()
    
    for K in Ks1:
        ndcgs[K]=[]
    for K in Ks2:
        aps[K]=[]
    df = pd.read_csv(test_path,engine="python")
    history=dict()
    # 排除训练集的影响
    df2=pd.read_csv(train_path,names=["userId","moviei","moviej","moviek"],engine="python")
    for uid, group in df2.groupby(["userId"]):
        ids=group.moviei.unique()
        history[uid]=ids
     
    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        ids = group.movieId.unique()
        # mAP的检测综合了所以样本，包括未观测的
        scores = []
        # 用于ndcg，只考虑测试集中的数据
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # 排除训练集的影响
        for i in history[uid]:
            scores[int(i)][1]=0
        
        for row, one in group.iterrows():
            u, i, r = [int(one["userId"]), int(one["movieId"]), one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks1:
            ndcgs[K].append(nDCG(K,scores2,method))
        for K in Ks2:
            aps[K].append(AP(K,scores,ids,method))
            
    for K in Ks1:
        print("model{0} NDCG@{1}:{2}".format(model.name, K, np.mean(ndcgs[K])))
    for K in Ks2:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))
    return ndcgs,aps


def pred_to_save(model,test_path,save_path):
    print("start ro save pred res...")
    df = pd.read_csv(test_path,engine="python")
    uIds=df.userId.tolist()
    iIds=df.movieId.tolist()
    ratings=df.rating.tolist()
    preds=[]
    for i in range(len(uIds)):
        preds.append(model.predict(uIds[i],iIds[i]))
    
    df2 = pd.DataFrame({"userId": uIds,
                       "movieId": iIds,
                       "rating": ratings,
                       "pred": preds})
    df2.to_csv(save_path,index=False)



def pred2save(model,test_path,save_path,index=1):
    """ index 用于指定模型返回结果中预测值的位置
        """
    print("start ro save pred res...")
    df = pd.read_csv(test_path,engine="python")
    uIds=df.userId.values
    iIds=df.movieId.values
    jIds=np.array([0]*len(uIds))
    ratings=df.rating.tolist()
    # preds=[]
    # for i in range(len(uIds)):
    #     preds.append(model.predict(uIds[i],iIds[i]))
    input_data={
        "user_input"  :uIds,
        "item_i_input":iIds,
        "item_j_input":jIds
    }
    res=model.predict(input_data)
    preds = [i[0]
             for i in res[index]
             ]


    df2 = pd.DataFrame({"userId": uIds,
                       "movieId": iIds,
                       "rating": ratings,
                       "pred": preds})
    df2.to_csv(save_path,index=False)




def ndcg_map_evulations_pred(model,Ks1,Ks2,train_path,test_path,n_itmes,method=1):

    aps=dict()
    ndcgs=dict()
    
    for K in Ks1:
        ndcgs[K]=[]
    for K in Ks2:
        aps[K]=[]
    df = pd.read_csv(test_path,engine="python")
    history=dict()
    # 排除训练集的影响
    df2=pd.read_csv(train_path,names=["userId","moviei","moviej","moviek"],engine="python")
    for uid, group in df2.groupby(["userId"]):
        ids=group.moviei.unique()
        history[uid]=ids
     
    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        ids = group.movieId.unique()
        # mAP的检测综合了所以样本，包括未观测的
        scores = []
        # 用于ndcg，只考虑测试集中的数据
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # 排除训练集的影响
        for i in history[uid]:
            scores[int(i)][1]=0
        
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks1:
            ndcgs[K].append(nDCG(K,scores2,method))
        for K in Ks2:
            aps[K].append(AP(K,scores,ids,method))
            
    for K in Ks1:
        print("model{0} NDCG@{1}:{2}".format(model.name, K, np.mean(ndcgs[K])))
    for K in Ks2:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))
    return ndcgs,aps




def ndcg_map_evulations_normal(model,Ks1,Ks2,train_path,test_path,n_itmes,method=1):

    # 对数据的处理不同
    aps=dict()
    ndcgs=dict()
    
    for K in Ks1:
        ndcgs[K]=[]
    for K in Ks2:
        aps[K]=[]
    df = pd.read_csv(test_path,engine="python")
    history=dict()
    # 排除训练集的影响
    # df2=pd.read_csv(train_path,names=["userId","moviei","moviej","moviek"],engine="python")
    df2=pd.read_csv(train_path,engine="python")
    for uid, group in df2.groupby(["userId"]):
        ids=group.movieId.unique()
        history[uid]=ids
     
    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        ids = group.movieId.unique()
        # mAP的检测综合了所以样本，包括未观测的
        scores = []
        # 用于ndcg，只考虑测试集中的数据
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # 排除训练集的影响
        # for i in history[str(uid)]:
        for i in history[uid]:
            scores[int(i)][1]=0
        
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks1:
            ndcgs[K].append(nDCG(K,scores2,method))
        for K in Ks2:
            aps[K].append(AP(K,scores,ids,method))
            
    for K in Ks1:
        print("model{0} NDCG@{1}:{2}".format(model.name, K, np.mean(ndcgs[K])))
    for K in Ks2:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))
    return ndcgs,aps



def ndcg_map_evulations2(model,Ks1,Ks2,train_path,test_path,n_itmes,method=0):

    aps=dict()
    ndcgs=dict()
    
    for K in Ks1:
        ndcgs[K]=[]
    for K in Ks2:
        aps[K]=[]
    df = pd.read_csv(test_path,engine="python")
    history=dict()
    # 排除训练集的影响
    df2=pd.read_csv(train_path,engine="python")
    for uid, group in df2.groupby(["userId"]):
        ids=group.movieId.unique()
        history[uid]=ids
    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        ids = group.movieId.unique()
        # mAP的检测综合了所以样本，包括未观测的
        scores = []
        # 用于ndcg，只考虑测试集中的数据
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # 排除训练集的影响
        for i in history[uid]:
            scores[i][1]=0
        
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks1:
            ndcgs[K].append(nDCG(K,scores2,method))
        for K in Ks2:
            aps[K].append(AP(K,scores,ids))
            
    for K in Ks1:
        print("model{0} NDCG@{1}:{2}".format(model.name, K, np.mean(ndcgs[K])))
    for K in Ks2:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))
    return ndcgs,aps


def ndcg_map_evulations_ALL(model,Ks1,Ks2,train_path,test_path,n_itmes):
    """
    包含了map和两个ndcg的评价
    ndcg分为包含所以未观测数据的和只包含测试集的两种
    """
    aps=dict()
    ndcgs=dict() #包含为观测数据
    ndcgs2=dict()
    for K in Ks1:
        ndcgs[K]=[]
    for K in Ks2:
        aps[K]=[]
    df = pd.read_csv(test_path,engine="python")
    history=dict()
    df2=pd.read_csv(train_path,names=["userId","moviei","moviej","moviek"],engine="python")
    for uid, group in df2.groupby(["userId"]):
        ids=group.moviei.unique()
        history[uid]=ids
    for uid, group in tqdm(df.groupby(["userId"]),ncols=50):
        ids = group.movieId.unique()
        scores = []
        scores2=[]
        for i in range(n_itmes):
            scores.append([i, model.predict(uid, i), 0])
        # 排除训练集的影响
        for i in history[uid]:
            scores[i][1]=0
        for row, one in group.iterrows():
            u, i, r = [one["userId"], one["movieId"], one["rating"]]
            scores[i][2]=r
            scores2.append(scores[i])
        shuffle(scores)
        for K in Ks1:
            ndcgs[K].append(nDCG(K,scores))
            ndcgs2[K].append(nDCG(K,scores2))
        for K in Ks2:
            aps[K].append(AP(K,scores,ids))
            
    for K in Ks1:
        print("model{0} NDCG@{1}:{2},NDCG2@{1}:{3}".format(
            model.name, K, np.mean(ndcgs[K]), np.mean(ndcgs2[K])))
    for K in Ks2:
        print("model{0} mAP@{1}:{2}".format(model.name, K, np.mean(aps[K])))
    return aps,ndcgs,ndcgs2


