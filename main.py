'''
第3.5代rnn
使用了kreas
预测准确率能达到80%
双向lstm构架
兼容GRU CNN
可保存实验数据到yaml文件
可从保存的模型继续训练
'''
import models.net as net
from models.db import Para
import sys
from models.mytools import *

def train_mode_get_his(model, his,para):
    X_train = get_pssm('data/train.npy')
    # X_train=get_onehot('data/train.npy')
    Y_train = get_dssp('data/train.npy')
    myHistory = his
    model.fit(X_train, Y_train,
              batch_size=para.batch,
              epochs=para.epoch,
              # validation_split=0.1,
              callbacks=[myHistory]
              )
def train(model,epochs=20,batch_size=64,info='default',id=0):
    X_train=get_pssm('data/train.npy')
    #X_train=get_onehot('data/train.npy')
    Y_train=get_dssp('data/train.npy')
    myHistory=Histories()
    model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            # validation_split=0.1,
            callbacks=[myHistory]
            )
    if SUMMARY:
        model.summary()
    model.save(MODEL_PATH)
    if RECORD :
        if id==0:
            id=np.random.randint(0,10000)
        write_history(myHistory,info,id=id)
        print('模型保存id',id)
    
    return id,myHistory
# 模型全局参数
SEQ_SIZE_MAX = 760
MODEL_SAVE = True
MODEL_PATH = 'data/saved_model.h5'
PLOT_TITLE='default'
SUMMARY=True
RECORD=False
PLOT = True
EPOCHS=20
LEARN_RATE=0.005
#####

def test(para):
    model=net.getNetByPara(para)
    id, myHistory = train(model, para.epoch, info=str(para.to_dict()))
    return myHistory
def testGru():
    import os
    model=gru_model_2(lr=LEARN_RATE)
    id,myHistory=train(model,EPOCHS,info='lr_0.005 gru dropout0.5   20_epochs onehot')
    if PLOT:
        if id!=0:
            plot_record(id)
        else:
            plot_history(myHistory)
def testCnn():
    import os
    model=cnn_model(lr=LEARN_RATE)
    id,myHistory=train(model,EPOCHS,info='lr_0.005 cnn ')
    if PLOT:
        if id!=0:
            plot_record(id)
        else:
            plot_history(myHistory)

if __name__ == "__main__":
    testCnn()
   
    
    