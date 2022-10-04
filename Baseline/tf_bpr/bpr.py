import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding,Input
from tensorflow.python.keras.backend import shape
import numpy as np
import pandas as pd



def get_model(n_users=20000,n_items=20000,dim=10):
    u_input=Input((1),name="user_input",dtype=tf.int32)
    i_input=Input((1),name="item_i_input",dtype=tf.int32)
    j_input=Input((1),name="item_j_input",dtype=tf.int32)
    u_emb_lay=Embedding(n_users,dim)
    i_emb_lay=Embedding(n_items,dim)
    u_emb=u_emb_lay(u_input)
    i_emb=i_emb_lay(i_input)
    j_emb=i_emb_lay(j_input)
    pred_i=tf.reduce_sum(tf.multiply(u_emb,i_emb),axis=2)
    pred_j=tf.reduce_sum(tf.multiply(u_emb,j_emb),axis=2)
    loss = tf.math.log(tf.nn.sigmoid(pred_i-pred_j))
    # loss=tf.negative(out)

    model=Model(inputs=[u_input,i_input,j_input],outputs=[loss,pred_i,pred_j])

    return model

def l1_loss(y_true, y_pred):
    return tf.reduce_sum(
        y_true-y_pred,axis=1
    )

def bpr_train(model=None,path="run/r3/train0.csv"):
    if model is None:
        model = get_model()
    model.compile(optimizer="Adam",loss=[l1_loss,],metrics=[l1_loss,])
    # train_data={
    #     "user_input"  :np.array([1,1,1]),
    #     "item_i_input":np.array([1,2,3]),
    #     "item_j_input":np.array([2,3,4]),
    # }
    
    df=pd.read_csv(path,names=["u",'i',"j","q"])
    train_data={
        "user_input"  :df.u.values,
        "item_i_input":df.i.values,
        "item_j_input":df.j.values,
    }
    train_label=np.array([0]*len(df)).reshape(-1)
    # train_data=pd.DataFrame(train_data)
    model.fit(train_data,train_label,epochs=10,verbose=2)
    # print(
    #     model.predict(train_data)
    # )
    return model



if __name__=="__main__":
    pass
    