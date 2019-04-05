'''
第三代rnn
使用了kreas
预测准确率能达到80%
双向lstm构架
可从保存的模型继续训练
'''
import numpy as np
import keras
from keras.models import Sequential, load_model,Model
from keras import optimizers, callbacks
from keras.layers import TimeDistributed, Dense, Activation, Dropout, LSTM
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
import keras.backend as K
from record import *
def plot_history(*historys):
    plt.title('Q3')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    infos=[]
    for history in historys:
        plt.plot(history.q3)
        infos.append(history.info)
    plt.legend(infos, loc='upper left')
    plt.show()
def plot_record(*ids):
    plt.title('Q3')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    infos=[]
    for id in ids:
        data=read_record(id)
        plt.plot(data['q3'])
        infos.append(data['info'])
    plt.legend(infos, loc='upper left')
    plt.show()

    # plt.title('loss')
    # plt.ylabel('loss rate')
    # plt.xlabel('epoch')
    # infos=[]
    # for id in ids:
    #     data=read_record(id)
    #     plt.plot(data['loss'])
    #     infos.append(data['info'])
    # plt.legend(infos, loc='upper left')
    # plt.show()

def get_pssm(path):
    ds = np.load(path).item()
    pssm = np.array(ds['pssm'])
    del ds
    num = len(pssm)
    ret = np.zeros((num, SEQ_SIZE_MAX, 21))  # 29=氨基酸数21+二级结构数8 序列最多为759个氨基酸
    for i in range(num):
        for j in range(SEQ_SIZE_MAX):
            if j < len(pssm[i]):
                ret[i, j, 0:20] = pssm[i][j]
            else:
                ret[i, j, 20] = 1

    return ret


def get_dssp_raw(path):
    ds = np.load(path).item()
    dssp = np.array(ds['dssp'])
    del ds
    return dssp


def get_dssp(path):
    ds = np.load(path).item()
    dssp = ds['dssp']
    del ds
    num = len(dssp)
    print(num)
    ret = np.zeros((num, SEQ_SIZE_MAX, 4))
    onehot_dict = {'E': 0, 'H': 1, '-': 2, ' ': 3}
    for i in range(num):
            dssp[i] = dssp[i]+' '*(SEQ_SIZE_MAX-len(dssp[i]))
            # print(dssp[i])
            # print(i,len(dssp[i]))
            for j in range(759):
                k = onehot_dict[dssp[i][j]]
                ret[i, j, k] = 1
        # 29=氨基酸数21+二级结构数8 序列最多为759个氨基酸
    return ret
# np.random.seed(100)


def dssp_trans(dssp=None, onehot=None):
    ret = []
    if onehot.any():
        for i in range(len(onehot)):
            s = ''
            for j in range(len(onehot[i])):
                if onehot[i, j][3] == 1:
                    break
                k = np.argmax(onehot[i, j])
                s += ['E', 'H', '-', ' '][k]
            ret.append(s)
    return ret


def Q3_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]):  # per aminoacid residue
            if real[i, j, 3] == 1:  # real[i, j, dataset.num_classes - 1] > 0 # if it is padding
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, 0:3])] > 0:
                    correct = correct + 1

    return correct / total


def q3_pred(y_true, y_pred):
    # q3=Q3_accuracy(y_true,y_pred)
    print(type(q3_pred))
    return K.mean(y_pred)


class Histories(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.loss = []
        self.q3 = []
        self.acc=[]
    def lr_change(self,epoch):
        if epoch ==10:
            K.set_value(self.model.optimizer.lr, .006)
        if epoch ==20:
            K.set_value(self.model.optimizer.lr, .005)  
        if epoch==30:
            K.set_value(self.model.optimizer.lr, .004)   
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        if GRAD:
            self.lr_change(epoch)
        return
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(float(logs.get('loss')))
        self.acc.append(logs.get('acc'))
        y_pred = self.model.predict(self.validation_data[0])
        y_real= self.validation_data[1]
        q3=Q3_accuracy(y_real, y_pred)
        self.q3.append(q3)
        print("Q3 accuracy: " + str(q3))
        
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return
    
def write_history(history,info='default',id=0):
    data={}
    data['info']=info
    data['q3']=history.q3
    data['loss']=history.loss
    add_record(id,data)
def get_info(args):
    pass

def model():
    
    model = Sequential ()
    
    # model.add( Conv1D (128,11,padding='same',activation='relu',input_shape=( 760 ,21 ) ) )
    # model.add(Dropout(0.2))
    # model.add(Conv1D(64, 11, padding='same', activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(4, 11, padding='same', activation='softmax'))
    

    opt = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['acc'])
    return model
############
def test_saved():
    X_test=get_pssm('test.npy')
    Y_test=get_dssp('test.npy')
    m=load_model(MODEL_PATH)
    predictions = m.predict(X_test,)
    print("\n\nQ3 accuracy: " + str(Q3_accuracy(Y_test, predictions)) + "\n\n")
    p=dssp_trans(onehot=predictions)
    y=get_dssp_raw('test.npy')
    print(p[1])
    print(y[1])
def train_from_saved():
    X_train=get_pssm('train.npy')
    Y_train=get_dssp('train.npy')
    myHistory=Histories()
    model=load_model(MODEL_PATH)
    model.fit(X_train, Y_train,
            batch_size=64,
            epochs=10,
            validation_split=0.1,
            callbacks=[myHistory])
    model.save(MODEL_PATH)
    return myHistory
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

def train(model,epochs=20,batch_size=64,info='default',id=0):
    X_train=get_pssm('train.npy')
    Y_train=get_dssp('train.npy')
    myHistory=Histories()
    model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[myHistory]
            )
    if SUMMARY:
        model.summary()
    model.save(MODEL_PATH)
    if RECORD :
        if id==0:
            id=np.random.randint(0,1000)
        write_history(myHistory,info,id=id)
        print('模型保存id',id)
    if PLOT:
        plot_record(id)
    return id
# 模型全局参数
SEQ_SIZE_MAX = 760
PLOT = True
MODEL_SAVE = True
MODEL_PATH = 'saved_model.h5'
PLOT_TITLE='default'
SUMMARY=False
RECORD=True
GRAD=False
def test():
    import os
    model=lstm_model(lr=0.007)
    train(model,40,info='lr0.007')
    
    os.system('plaympeg 1.mp3')
if __name__ == "__main__":
    plot_record(157,16,801)
    # test()
    