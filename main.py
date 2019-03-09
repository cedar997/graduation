import tensorflow as tsf
import numpy as np 
import readdata
from keras.models import Sequential
from keras.layers import MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras import regularizers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix

def the_nn(X, Y, val_data, epochs, batch_size, node1, node2, reg, opti, filter1, filter2):
    model = Sequential() 
    model.add(Conv2D(node1, filter1, input_shape=(X.shape[1], X.shape[2], 1), padding='same', activation='relu',
                     kernel_regularizer=reg))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(node2, filter2, activation='relu', padding='same',
                     kernel_regularizer=reg))
    model.add(Flatten())
    model.add(Dense(Y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    #X 81381*20*21*1
    #Y 81381*3
    history = model.fit(X, Y, validation_data = val_data, epochs = epochs, batch_size = batch_size)
    return model, history
if __name__ == "__main__":
    ##训练阶段
    train_seq,train_pssm,train_dssp=readdata.load_file('train.npy')
    test_seq,test_pssm,test_dssp=readdata.load_file('test.npy')
    X=readdata.format_pssm(train_pssm)
    Y=readdata.format_dssp(train_dssp)
    teX=readdata.format_pssm(test_pssm)
    teY=readdata.format_dssp(test_dssp)
    ##参数
    BATCH_SIZE=1000
    EPOCHS=50
    OPTIMIZER="adam"
    ##模型
    model, history = the_nn(X, Y, (teX,teY), EPOCHS, BATCH_SIZE, 96, 
                                    11, regularizers.l1(0.01), OPTIMIZER, (5, 5), (2, 2))
    ###结果
    
    
    
    