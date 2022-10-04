import os
from subprocess import run

import numpy as np

from until.evulation import (ndcg_evaluation, pr_evulation, pr_ndcg_evulation,
                             pr_ndcg_evulation_All)


class Rand:
    name="Rand"
    def predict(self,u,i):
        return np.random.random()

class Simi:
    name="Simi"
  
    def __init__(self,method,n_users,n_items,dim,train_path,lr=0.007,rg=0.05,iter=20):
        self.method=method
        self.n_users=n_users
        self.n_items=n_items
        self.dim=dim
        self.train_path=train_path
        self.pu=np.zeros((n_users,dim))
        self.qi=np.zeros((n_items,dim))
        self.lr=lr
        self.rg=rg
        self.iter=iter
    def predict(self,uid,iid):
        return np.dot(self.pu[uid],self.qi[iid])

    def fit(self,u_path,i_path):
        # u_path="u.bin"
        # i_path="i.bin"
        exe=r"E:\设计文件\joint\model.exe"
        cmd = "{} -Method {} -n_users {} -n_items {} -dim {} -lr {} -rg {} -iter {} \
            -train_path {} -u_path {} -i_path {}".format(
            exe, self.method,self.n_users, self.n_items, self.dim,
            self.lr, self.rg, self.iter,
            self.train_path, u_path, i_path
        )
        # os.system(cmd)
        run(cmd)
        self.load(u_path,i_path)
        

    def load(self,u_path,i_path):
        self.pu = np.fromfile(u_path)
        self.pu.shape = self.n_users, self.dim
        self.qi = np.fromfile(i_path)
        self.qi.shape = self.n_items, self.dim


if __name__ == "__main__":
    # a=Rand()
    # evaluation(a,10,"test_file.csv")
    
    n_users=6400
    n_itmes=4000
    model=Simi(6400,4000,10,"train_file.csv")
    # model.fit()
    
    
    print("hello,world!")
    model.load("NDCGu.bin","NDCGi.bin")
    # ndcg_evaluation(model,5,"NDCG\\test_file.csv")
    # model.load("PRu.bin","PRi.bin")
    # pr_evulation(model,10,"PR\\test_file.csv","PR\\test_ng.dict")
    # pr_ndcg_evulation(model,10,"PR\\test_file.csv","PR\\test_ng.dict")
    # pr_ndcg_evulation_All(model,5,"PR\\test_file.csv",n_itmes)
    pr_ndcg_evulation_All(model,5,"NDCG\\test_file.csv",n_itmes)
