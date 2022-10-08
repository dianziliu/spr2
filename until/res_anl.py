import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from math import log2
from tqdm import tqdm
from evulation import nDCG
import matplotlib.pyplot as plt

def DCG(K, l, w,method=0):
    # [[iId,score,rating],...]
    s = sorted(l, key=lambda x: x[1], reverse=True)
    if method:
        x=[i[2] for i in s[:K]]
    else:
        x=[pow(2,i[2])-1 for i in s[:K]] #fixed this bug
    return np.dot(x, w[:K]),s[:K]

def iDCG(K, l, w,method=0):
    s = sorted(l, key=lambda x: x[2], reverse=True)
    if method:
        x=[i[2] for i in s[:K]]
    else:
        x=[pow(2,i[2])-1 for i in s[:K]]
    return np.dot(x, w),s[:K]

def nDCG_l(K, l,method=0):
    """
    k:int
    l:list[(int ,int ,int)]
    id,pred_score,rating
    """
    if len(l)<K:
        K=len(l)
    w = [1/log2(2+i) for i in range(K)]
    dcg,rank_dcg = DCG(K, l, w,method)
    idcg,rank_idcg = iDCG(K, l, w,method)
    return dcg/idcg,rank_dcg,rank_idcg




def ndcg_evulations_pred(Ks1,test_path,src_data,key="userId",method=1):
    
    list_u     = []
    dict_len   = dict()
    dict_group = dict()

    df = pd.read_csv(test_path, engine="python")

    # 用户划分，可能需要针对具体数据集进行定制
    User_Group = {"<32": [],
                  "32-64": [],
                  "64-128": [],
                  "128-256": [],
                  "256-512": [],
                  "512-1024": [],
                  ">1024": []}

    df_src = pd.read_csv(src_data)

    # 将每一个用户进行切分，并进行统计
    for uid, group in df.groupby([key]):
        list_u.append(uid)
        # dict_len[u] = len(group)*5
        dict_group[uid] = group

    # 统计每个用户的长度
    for uid, group in df_src.groupby([key]):
        dict_len[uid] = len(group)

    # 划分用户组
    for u, u_ratings in dict_len.items():
        if u_ratings < 32:
            User_Group["<32"].append(u)
        elif u_ratings < 64:
            User_Group["32-64"].append(u)
        elif u_ratings < 128:
            User_Group["64-128"].append(u)
        elif u_ratings < 256:
            User_Group["128-256"].append(u)
        elif u_ratings < 512:
            User_Group["256-512"].append(u)
        elif u_ratings < 1024:
            User_Group["512-1024"].append(u)
        else:
            User_Group[">1024"].append(u)

    dict_evl = {
        "group_name": [],
        "count_user": [],
        "count_rating": []
    }
    for k in Ks1:
        dict_evl["ndcg@{}".format(k)]=[]

    for group_name, group in User_Group.items():
        lables = []
        preds = []
        ndcgs = dict()
        count_l=0
        for K in Ks1:
            ndcgs[K] = []

        if len(group) == 0:
            continue
        for one in group:
            # dict_evl["list_user"].append(one)
            if one not in dict_group:
                continue
            one    = dict_group[one]
            iids=one.movieId.to_list()
            labels = one.rating.to_list()
            preds  = one.pred.to_list()
            count_l+=len(preds)
            res = [(iid, label, pred)
                   for iid, label, pred in zip(iids,preds,labels)]
            for k in Ks1:
                ndcgs[k].append(nDCG(k, res, method))
        dict_evl["group_name"].append(group_name)
        dict_evl["count_user"].append(len(group))
        dict_evl["count_rating"].append(count_l)
        for k in Ks1:
            dict_evl["ndcg@{}".format(k)].append(np.mean(ndcgs[k]))
    # count_ndcgs=[]
    # for uId,group in df.groupby(["userId"]):
    #     iids=group.movieId.to_list()
    #     ratings=group.rating.to_list()

    return dict_evl

def ndcg_evulations_pred2(Ks1,n_items,test_path,key="userId",method=1):
    

    best_hit_item=dict()
    for i in range(n_items):
        best_hit_item[i]=0  
    pred_hit_item=dict()
    for i in range(n_items):
        pred_hit_item[i]=0  

    df = pd.read_csv(test_path, engine="python")

    K=10
    for uid,group in df.groupby(key):
        iids=group.movieId.to_list()
        labels = group.rating.to_list()
        preds  = group.pred.to_list()
        res = [(iid, label, pred)
                for iid, label, pred in zip(iids,preds,labels)]
        ndcg,rank_dcg,rank_idcg=nDCG_l(K,res)
        
        for one in rank_dcg:
            pred_hit_item[one[0]]+=1
        for one in rank_idcg:
            best_hit_item[one[0]]+=1
    x1=list(pred_hit_item.keys())
    y1=list(pred_hit_item.values())
    y1=sorted(y1,reverse=True)

    plt.plot(x1,y1)
    plt.show()
    return pred_hit_item,best_hit_item



def analy_users(K,test_path,src_data,method=1):
    
    
    dict_len   = dict()
    
    df = pd.read_csv(test_path, engine="python")
    df_src = pd.read_csv(src_data)



    # 统计每个用户的长度
    for iid, group in df_src.groupby(["movieId"]):
        dict_len[iid] = len(group)

    ndcgs      = []
    rank_dcgs  = []
    rank_idcgs = []

    for uid, group in df.groupby(["userId"]):

        iids   = group.movieId.to_list()
        labels = group.rating.to_list()
        preds  = group.pred.to_list()
        res = [(iid, label, pred)
                for iid, label, pred in zip(iids,preds,labels)]
        ndcg,rank_dcg,rank_idcg=nDCG_l(K,res,method=1)
        ndcgs.append(ndcg)
        rank_dcgs.append(rank_dcg)
        rank_idcgs.append(rank_idcg)
    
    mean_ndcg=np.mean(ndcgs)
    # best_uid=np.argmin(np.abs(np.array(ndcgs)-mean_ndcg))

    best_uids=np.argsort(np.abs(np.array(ndcgs)-mean_ndcg))[:5]
    for best_uid in best_uids:
        best_dcg=rank_dcgs[best_uid]
        best_idcg=rank_idcgs[best_uid]

        items_dcg  = [i[0] for i in best_dcg]
        items_idcg = [i[0] for i in best_idcg]
        r_dcg      = [i[2] for i in best_dcg]
        r_idcg     = [i[2] for i in best_idcg]
        fre_dcg    = [dict_len[i[0]] for i in best_dcg]
        fre_idcg   = [dict_len[i[0]] for i in best_idcg]

        eval = pd.DataFrame({"dcg_items": items_dcg,
                            "rating_dcg": r_dcg,
                            "fre_dcg": fre_dcg,
                            "idcg_items": items_idcg,
                            "rating_idcg": r_idcg,
                            "fre_idcg": fre_idcg}
                            )
        print(best_uid,ndcgs[best_uid],np.mean(fre_dcg),np.mean(fre_idcg))
        yield best_uid,eval

    # return best_uid,eval



def gruop_analy_users():

    Path1=["res_save/SPR_ml100k_{}.csv",
    "res_save/SPR_ml1m_{}.csv",
    "res_save/SPR_r3_{}.csv",
    "res_save/SPR_r4_{}.csv"
    ]
    Path2=["data/Ml100Krating.csv",
    "data/ML1Mratings.csv",
    "data/YahooR3.csv",
    "data/YahooR4.csv"
    ]
    for path1_f,path2 in zip(Path1,Path2):
        for i in range(5):
            path1=path1_f.format(i)
            
            uid,eval=analy_users(10,path1,path2)
            # eval.to_csv("res_save/test.csv",sep="\t")
            print(path1,uid)
            print(eval)
    
if __name__=="__main__":
    print("start to analy")
    # gruop_analy_users()

    path1="res_save/SPR_ml100k_4.csv"
    path2="data/Ml100Krating.csv"
    ndcg_evulations_pred2([10],1800,path1)

    # print(pd.DataFrame(
    #     ndcg_evulations_pred([5, 10], path1, path2)
    #     ))
    # for uid,eval in analy_users(10,path1,path2):

    # # eval.to_csv("res_save/test.csv",sep="\t")
    #     print(eval)
