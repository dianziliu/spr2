#Version 1.0
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import random
import pickle
from decimal import *
import numpy as np
import time 
# Global variables
__author__ = 'Arthur Fortes'
learn_rate = 0.01
regularization = 0.01
reg_u = 0.01
reg_i = 0.01
reg_j = 0.01
getcontext().prec = 20
number_int = 15000
rating_max=5
rating_min=1
alpha=0.1
# Reading training data, and obtain the rated item set and the unrated item set of all users
def read_file_without_scores(file_to_read, space_type='\t'):
    dict_items = dict()
    dict_not_items = dict()
    list_users = list()
    list_items = list()
    cont = 0
    fi=open(file_to_read, 'r')
    fi.readline()
    for line in fi:
        if line.strip():
            cont += 1
            inline = line.split(space_type)
            list_users.append(int(inline[0]))     
            list_items.append(int(inline[1]))
            if int(inline[0]) in dict_items:
                dict_items[int(inline[0])] += [int(inline[1])]
            else:
                dict_items[int(inline[0])] = [int(inline[1])]     
    fi.close()
    list_users = sorted(list(set(list_users)))    
    list_items = sorted(list(set(list_items)))    
    for user in list_users:
        dict_not_items[user] = list(set(list_items) - set(dict_items[user]))        
    return dict_items, dict_not_items, list_users, list_items, cont, len(list_users), len(list_items)

# Initialize the feature matrices for users and items
def create_factors(num_users, num_items, factors):
    users_factors = np.random.uniform(0, 1, [num_users, factors])
    items_factors = np.random.uniform(0, 1, [num_items, factors])
    return users_factors, items_factors

# Define parameters and attributes in RBPR algorithm

class RBPR(object):
    def __init__(self, file_to_train, file_to_write, space_type='\t', num_factors=10, num_interactions=1000):
        self.file_to_train = file_to_train
        self.file_to_write = file_to_write
        self.num_factors = num_factors
        self.num_interactions = num_interactions
        dict_items, dict_not_items, list_users, list_items, num_int, num_users, num_items = read_file_without_scores(
            file_to_train, space_type)
        self.dict_items = dict_items
        self.dict_not_items = dict_not_items
        self.list_users = list_users
        self.list_items = list_items
        self.num_int = num_int
        self.num_users = num_users
        self.num_items = num_items
        self.u_factors, self.i_factors = create_factors(num_users, num_items, num_factors)
        print("Training data: " + str(num_users) + " users | " + str(num_items) + " items | " +
              str(num_int) + " interactions...")
        print("RBPR num_factors=" + str(num_factors) + " | reg_u=" + str(reg_u) +
              " | reg_i=" + str(reg_i) + " | reg_j=" + str(reg_j) + " | learn rate= " + str(learn_rate))
        
    # Construct the input item pair of RBPR   
    def sample_triple(self):
        user = random.choice(self.list_users)
        item = random.choice(self.dict_items[user])
        other_item = random.choice(self.dict_not_items[user])
        user_id = self.list_users.index(user)
        item_id = self.list_items.index(item)
        other_item_id = self.list_items.index(other_item)
        return user, item, other_item, user_id, item_id, other_item_id
    
    # Trian RBPR
    def train_rbpr(self):
        rmse_result=[]
        time_res=[]
        for i in range(self.num_interactions):
            # self.iterate_rbpr()
            a=time.time()
            self.iterate_rbpr()
            b=time.time()
            time_res.append(b-a)
        print("mean iter use {} second".format(np.mean(time_res)))            
    # Iterative process of RBPR
    def iterate_rbpr(self):
        i = 0
        for _ in range(number_int):
            i += 1
            user,item, other_item, user_id, item_id, other_item_id = self.sample_triple()
            self.update_factors_rbpr(user, item, other_item, user_id, item_id, other_item_id)
        return self.u_factors, self.i_factors
    
    def predict(self,user,item):
        try:
            u = self.list_users.index(user)
            i = self.list_items.index(item)
        except ValueError as e:
            return 0
            
        rui = sum(np.array(self.u_factors[u]).T * np.array(self.i_factors[i]))
        return rui

    # Updating method of user and item feature vectors in RBPR
    def update_factors_rbpr(self, user, item, other_item, u, i, j):
        rui = sum(np.array(self.u_factors[u]).T * np.array(self.i_factors[i]))
        ruj = sum(np.array(self.u_factors[u]).T * np.array(self.i_factors[j]))
        A=rating_max   #the maximum value of user score in the training set
        B=rating_min   #the minimum value of user score in the training set
        K2 = alpha
        K1 = 1-alpha
        
        # eui1 = V[uid,iid]
        # ruj0 = P_rating[user-1,other_item-1]
        # ruj0 = pre_otheritem_score #the predicted score rated by user u on other_item through SVD algorithm
        
        eui1=1
        ruj0=1
        
        x_uij =(rui - ruj)
        temp=math.exp(-x_uij)
        fun_exp = float(temp) / float((1.0 + temp))
        g_ui = 1/float((1.0 +math.exp(-rui)))
        delta_gui = math.exp(-rui)*g_ui*g_ui
        g_uj = 1/float((1.0 +math.exp(-ruj)))
        delta_guj = math.exp(-ruj)*g_uj*g_uj
        for num in range(self.num_factors):
            w_uf = self.u_factors[u][num]
            h_if = self.i_factors[i][num]
            h_jf = self.i_factors[j][num]
            update_user = K1*(h_if - h_jf) * fun_exp - reg_u * w_uf + K2*0.5*A*(eui1+B-A*g_ui)*delta_gui* h_if + K2*0.5*A*(ruj0+B-A*g_uj)*delta_guj* h_jf
            self.u_factors[u][num] = w_uf + learn_rate * update_user
            update_item_i =K1*w_uf * fun_exp - reg_i * h_if + K2*0.5*A*(eui1+B-A*g_ui)*delta_gui * w_uf      
            self.i_factors[i][num] = h_if + learn_rate * update_item_i
            update_item_j = -K1*w_uf * fun_exp - reg_j * h_jf + K2*0.5*A*(ruj0+B-A*g_uj)*delta_guj * w_uf
            self.i_factors[j][num] = h_jf + learn_rate * update_item_j