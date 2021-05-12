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
from keras.models import Sequential, load_model, Model
from keras import optimizers, callbacks
from keras.layers import TimeDistributed, Dense, Activation, Dropout, LSTM, GRU, Conv1D
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
import keras.backend as K
from models.mytools import *
import sys

'''
cnn为卷积神经网络
'''


def cnn_model(lr=0.0005, input_dim=21, output_dim=4):
    m = Sequential()
    m.add(Conv1D(128, 11, padding='same', activation='relu', input_shape=(760, 21)))
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


def lstm_model(unit=128, lr=0.008, input_dim=21, output_dim=4):
    model = Sequential()
    model.add(Bidirectional(LSTM(
        unit, return_sequences=True,
        input_shape=(None, input_dim)
    )))
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


# 两层的gru
def gru_model_2(unit=128, lr=0.008, input_dim=21, output_dim=4):
    model = Sequential()
    model.add(Bidirectional(GRU(
        unit, return_sequences=True,
        input_shape=(None, input_dim)
    )))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(unit, return_sequences=True)))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


# 三层的gru
def gru_model_3(unit=128, lr=0.008, input_dim=21, output_dim=4):
    model = Sequential()
    model.add(Bidirectional(GRU(
        unit, return_sequences=True,
        input_shape=(None, input_dim)
    )))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(unit, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(unit, return_sequences=True)))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


# 双向的单层gru
def gru_model(unit=128, lr=0.008, input_dim=21, output_dim=4):
    model = Sequential()
    model.add(Bidirectional(GRU(
        unit, return_sequences=True,
        input_shape=(None, input_dim)
    )))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


# 单层单向gru
def gru_model_uni(unit=128, lr=0.008, input_dim=21, output_dim=4):
    model = Sequential()
    model.add((GRU(
        unit, return_sequences=True,
        input_shape=(None, input_dim)
    )))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model

net_map={
    'cnn_model':'卷积神经网络',
    'lstm_model': 'lstm-单层双向',
    'gru_model':'单层双向gru',
    'gru_model_2':'双层双向gru',
    'gru_model_uni':'单层单向gru',
    'gru_model_3':'三层双向gru'
}


#通过参数获得网络
def getNetByPara(para):
    fn_obj = getattr(sys.modules[__name__], para.net)
    return fn_obj(lr=para.lr)





