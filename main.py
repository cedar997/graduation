'''
第3.5代rnn
使用了kreas
预测准确率能达到80%
双向lstm构架
兼容GRU CNN
可保存实验数据到yaml文件
可从保存的模型继续训练
'''
import numpy as np
import keras
from keras.models import Sequential, load_model,Model
from keras import optimizers, callbacks
from keras.layers import TimeDistributed, Dense, Activation, Dropout, LSTM,GRU,Conv1D
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
import keras.backend as K
from mytools import *
'''
cnn为卷积神经网络
'''
def cnn_model(lr=0.0005,input_dim=21,output_dim=4):
    m = Sequential ()
    m.add( Conv1D (128,11,padding='same',activation='relu',input_shape=( 760 ,21 ) ) )
    m.add(Dropout(0.2))
    m.add(Conv1D(64, 11, padding='same', activation='relu'))
    m.add(Dropout(0.2))
    m.add(Conv1D(4, 11, padding='same', activation='softmax'))
    opt = optimizers.Adam(lr=lr)
    m.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['acc'])
    m.summary()
    return m
'''
长短期记忆网络（LSTM，Long Short-Term Memory）
是一种时间循环神经网络，
是为了解决一般的RNN（循环神经网络）
存在的长期依赖问题而专门设计出来的
'''
def lstm_model(unit=128,lr=0.008,input_dim=21,output_dim=4):
    model=Sequential()
    model.add( Bidirectional( LSTM(
                 unit,return_sequences=True,
                input_shape=(None,input_dim)
             )               )      )
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model

'''
GRU是LSTM网络的一种效果很好的变体，
它较LSTM网络的结构更加简单，而且效果也很好，
因此也是当前非常流形的一种网络。
可以解决RNN网络中的长依赖问题。
'''
#两层的gru
def gru_model_2(unit=128,lr=0.008,input_dim=21,output_dim=4):
    model=Sequential()
    model.add( Bidirectional( GRU(
                 unit,return_sequences=True,
                input_shape=(None,input_dim)
             )               )      )
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(unit,return_sequences=True)))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model
#三层的gru
def gru_model_3(unit=128,lr=0.008,input_dim=21,output_dim=4):
    model=Sequential()
    model.add( Bidirectional( GRU(
                 unit,return_sequences=True,
                input_shape=(None,input_dim)
             )               )      )
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(unit,return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(unit,return_sequences=True)))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model
#双向的单层gru
def gru_model(unit=128,lr=0.008,input_dim=21,output_dim=4):
    model=Sequential()
    model.add( Bidirectional( GRU(
                 unit,return_sequences=True,
                input_shape=(None,input_dim)
             )               )      )
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model
#单层单向gru
def gru_model_uni(unit=128,lr=0.008,input_dim=21,output_dim=4):
    model=Sequential()
    model.add( ( GRU(
                 unit,return_sequences=True,
                input_shape=(None,input_dim)
             )               )      )
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model
'''

'''
def train(model,epochs=20,batch_size=64,info='default',id=0):
    # X_train=get_pssm('train.npy')
    X_train=get_onehot('train.npy')
    Y_train=get_dssp('train.npy')
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
MODEL_PATH = 'saved_model.h5'
PLOT_TITLE='default'
SUMMARY=True
RECORD=False
PLOT = True
EPOCHS=20
LEARN_RATE=0.005
#####
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
   
    
    