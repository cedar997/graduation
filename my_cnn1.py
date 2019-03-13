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
    num=len(pssm)
    ret=np.zeros((num,SEQ_SIZE_MAX,20))    #29=氨基酸数21+二级结构数8 序列最多为759个氨基酸
    #ret[ : , : , 0 : 20 ]=pssm[ : , : , 0 : 20 ]   #载入pssm
    for i in range(num):
        for j in range(len(pssm[i])):
            ret[i,j,0:20]=pssm[i][j]
    return ret
def get_dssp(path):
    ds=np.load(path).item()
    dssp=np.array(ds['dssp'])
    num=len(dssp)
    ret=np.zeros((num,SEQ_SIZE_MAX,3))
    for i in range(len(dssp)):
            for j in range(len(dssp[i])):
                if dssp[i][j]=='E':
                    ret[i,j,0]=1
                elif dssp[i][j]=='H':
                    ret[i,j,1]=1
                else:
                    ret[i,j,2]=1
        #29=氨基酸数21+二级结构数8 序列最多为759个氨基酸
    return ret
#np.random.seed(100)
def dssp_trans(dssp=None,onehot=None):
    ret=[]
    if onehot.any():
        for i in range(len(onehot)):
            s=''
            for j in range(len(onehot[i])):
                if np.sum(onehot[i,j])==0:
                    break
                k=np.argmax(onehot[i,j])
                s+=['E','H','-'][k]
            ret.append(s)
    return ret
def Q3_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]): # per aminoacid residue
            if np.sum(real[i, j, :]) == 0:  #  real[i, j, dataset.num_classes - 1] > 0 # if it is padding
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
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
    m.add( Conv1D (128,11,padding='same',activation='relu',input_shape=( 760 ,20 ) ) )
    m.add(Dropout(0.3))
    m.add(Conv1D(64, 11, padding='same', activation='relu'))
    m.add(Dropout(0.3))
    m.add(Conv1D(3, 11, padding='same', activation='softmax'))
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
def test():
    X_train=get_pssm('train.npy')
    Y_train=get_dssp('train.npy')
    X_test=get_pssm('test.npy')
    Y_test=get_dssp('test.npy')
    m=model()
    history=m.fit( X_train,Y_train,epochs=100,batch_size=64,validation_split=0.1)
    predictions = m.predict(X_test)
    print("\n\nQ3 accuracy: " + str(Q3_accuracy(Y_test, predictions)) + "\n\n")
    if MODEL_SAVE==True:
        m.save(MODEL_PATH,True,True)
    if PLOT==True:
        plot_history(history)

if __name__ == "__main__":
    test()