##开始开发我的第一代卷积神经网络
import tensorflow as tf 
import numpy as np 
import readdata
###参数
BATCH_SIZE = 100 


###
def  train(X,Y,val_data):
    ##x为
    x=tf.placeholder(tf.float32,[
        BATCH_SIZE,X.shape[1],X.shape[2],X.shape[3]
    ],name="x-input")
    pass
if __name__ == "__main__":
    train_seq,train_pssm,train_dssp=readdata.load_file('train.npy')
    test_seq,test_pssm,test_dssp=readdata.load_file('test.npy')
    X=readdata.format_pssm(train_pssm)
    Y=readdata.format_dssp(train_dssp)
    teX=readdata.format_pssm(test_pssm)
    teY=readdata.format_dssp(test_dssp)
    