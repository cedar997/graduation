import numpy as np
from keras.models import Sequential,load_model
from keras import optimizers, callbacks
from keras.layers import Dense, Activation, Dropout, Conv1D
import matplotlib.pyplot as plt
##模型全局参数
SEQ_SIZE_MAX=760
PLOT=True
MODEL_SAVE=True
MODEL_PATH='saved_model.h5'
#####
def get_pssm(path):
    ds=np.load(path).item()
    pssm=np.array(ds['pssm'])
    del ds
    num=len(pssm)
    ret=np.zeros((num,SEQ_SIZE_MAX,21))    #29=氨基酸数21+二级结构数8 序列最多为759个氨基酸
    for i in range(num):
        for j in range(SEQ_SIZE_MAX):
            if j<len(pssm[i]):
                ret[i,j,0:20]=pssm[i][j]
            else:
                ret[i,j,20]=1

    return ret
def get_dssp_raw(path):
    ds=np.load(path).item()
    dssp=np.array(ds['dssp'])
    del ds
    return dssp
def get_dssp(path):
    ds=np.load(path).item()
    dssp=ds['dssp']
    del ds
    num=len(dssp)
    print(num)
    ret=np.zeros((num,SEQ_SIZE_MAX,4))
    onehot_dict={'E':0,'H':1,'-':2,' ':3}
    for i in range(num):
            dssp[i]=dssp[i]+' '*(SEQ_SIZE_MAX-len(dssp[i]))
            # print(dssp[i])
            # print(i,len(dssp[i]))
            for j in range(759):
                k=onehot_dict[dssp[i][j]]
                ret[i,j,k]=1
        #29=氨基酸数21+二级结构数8 序列最多为759个氨基酸
    return ret
#np.random.seed(100)
def dssp_trans(dssp=None,onehot=None):
    ret=[]
    if onehot.any():
        for i in range(len(onehot)):
            s=''
            for j in range(len(onehot[i])):
                if onehot[i,j][3]==1:
                    break
                k=np.argmax(onehot[i,j])
                s+=['E','H','-',' '][k]
            ret.append(s)
    return ret
def Q3_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]): # per aminoacid residue
            if real[i, j, 3] == 1:  #  real[i, j, dataset.num_classes - 1] > 0 # if it is padding
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, 0:3])] > 0:
                    correct = correct + 1

    return correct / total

def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for error
    
def model():
    m = Sequential ()
    m.add( Conv1D (128,11,padding='same',activation='relu',input_shape=( 760 ,21 ) ) )
    m.add(Dropout(0.3))
    m.add(Conv1D(64, 11, padding='same', activation='relu'))
    m.add(Dropout(0.3))
    m.add(Conv1D(4, 11, padding='same', activation='softmax'))
    opt = optimizers.Adam(lr=0.0005)
    m.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return m
def test_saved():
    X_test=get_pssm('test.npy')
    Y_test=get_dssp('test.npy')
    m=load_model(MODEL_PATH)
    predictions = m.predict(X_test)
    print("\n\nQ3 accuracy: " + str(Q3_accuracy(Y_test, predictions)) + "\n\n")
    p=dssp_trans(onehot=predictions)
    y=get_dssp_raw('test.npy')
    print(p[1])
    print(y[1])
def test():
    X_train=get_pssm('train.npy')
    Y_train=get_dssp('train.npy')
    X_test=get_pssm('test.npy')
    Y_test=get_dssp('test.npy')
    m=model()
    history=m.fit( X_train,Y_train,epochs=50,batch_size=64,validation_split=0.1)
    predictions = m.predict(X_test)
    print("\n\nQ3 accuracy: " + str(Q3_accuracy(Y_test, predictions)) + "\n\n")
    if MODEL_SAVE==True:
        m.save(MODEL_PATH,True,True)
    if PLOT==True:
        plot_history(history)

if __name__ == "__main__":
    # test()
    test_saved()