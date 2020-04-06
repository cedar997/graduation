import yaml
import numpy as np
import keras
from keras.models import Sequential, load_model,Model
from keras import optimizers, callbacks
from keras.layers import TimeDistributed, Dense, Activation, Dropout, LSTM,GRU,Conv1D
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
import keras.backend as K
###
file_name='a.yaml'
SEQ_SIZE_MAX = 760
MODEL_PATH = 'saved_model.h5'
GRAD=False
#是否将训练集的一部分用作验证
VALID=False
###
def plot_history(*historys):
    plt.title('Q3准确率随代数的变化')
    plt.ylabel('Q3准确率')
    plt.xlabel('代数')
    infos=[]
    
    for history in historys:
        plt.plot(history.q3)
        # infos.append(history.info)
    #plt.legend(infos, loc='upper left',fontproperties='SimHei')
    plt.legend(infos, loc='upper left')
    plt.show()
def plot_record(*ids,info=[]):
    xi=[i for i in range(20)]
    # plt.title('Q3准确率随代数的变化',fontproperties='SimHei')
    plt.ylabel('Q3准确率',fontproperties='SimHei')
    plt.xlabel('代数',fontproperties='SimHei')
    plt.xticks(xi)
    plt.axis([0,20,0.7,0.85])
    infos=info
    for id in ids:
        data=read_record(id)
        plt.plot(data['q3'])
        # infos.append(data['info'])
    plt.legend(infos, loc='upper left',prop={'family':'SimHei','size':14})
    plt.show()

    plt.title('loss')
    plt.ylabel('loss rate')
    plt.xlabel('epoch')
    infos=[]
    for id in ids:
        data=read_record(id)
        plt.plot(data['loss'])
        infos.append(data['info'])
    plt.legend(infos, loc='upper left')
    plt.show()
def get_seq(path):
    ds = np.load(path,allow_pickle=True).item()
    seq = np.array(ds['seq'])
    return seq
def get_pssm_20(path):
    ds = np.load(path,allow_pickle=True).item()
    pssm = np.array(ds['pssm'])
    del ds
    num = len(pssm)
    ret = np.zeros((num, SEQ_SIZE_MAX, 20))  # 29=氨基酸数21+二级结构数3 序列最多为759个氨基酸
    for i in range(num):
        for j in range(SEQ_SIZE_MAX):
            if j < len(pssm[i]):
                ret[i, j, 0:20] = pssm[i][j]
            else:
                ret[i, j] = [0]*20

    return ret
def get_pssm(path):
    ds = np.load(path,allow_pickle=True).item()
    pssm = np.array(ds['pssm'])
    del ds
    num = len(pssm)
    ret = np.zeros((num, SEQ_SIZE_MAX, 21))  # 29=氨基酸数21+二级结构数3 序列最多为759个氨基酸
    for i in range(num):
        for j in range(SEQ_SIZE_MAX):
            if j < len(pssm[i]):
                ret[i, j, 0:20] = pssm[i][j]
            else:
                ret[i, j, 20] = 1

    return ret
def get_onehot(path):
    ds = np.load(path,allow_pickle=True).item()
    seq = np.array(ds['seq'])
    del ds
    num = len(seq)
    ret = np.zeros((num, SEQ_SIZE_MAX, 21))
    aa_dict='ARNDCQEGHILKMFPSTWYV'
    for i in range(num):
        for j in range(SEQ_SIZE_MAX):
            if j < len(seq[i]):
                ind=aa_dict.find(seq[i][j])
                ret[i, j, ind] = 1
            else:
                ret[i, j, 20] = 1
    return ret
def get_dssp_raw(path):
    ds = np.load(path,allow_pickle=True).item()
    dssp = np.array(ds['dssp'])
    del ds
    return dssp


def get_dssp(path):
    ds = np.load(path,allow_pickle=True).item()
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

def Q3_EHC(real,pred):
    total = real.shape[0] * real.shape[1]
    correct = [0,0,0]
    ss_num=[0,0,0] #预测的二级结构类别的数目
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]):  # per aminoacid residue
            if real[i, j, 3] == 1:  # real[i, j, dataset.num_classes - 1] > 0 # if it is padding
                total = total - 1
            else:
                pred_ss=np.argmax(pred[i, j, 0:3])
                ss_num[pred_ss]+=1
                if real[i, j, pred_ss] > 0:
                    correct [pred_ss]+=1
    print(correct,ss_num,total)
    correct_rate=np.divide(correct,ss_num)
    return correct_rate
def SOV_EHC(real,pred):
    total = real.shape[0] * real.shape[1]
    correct = [0,0,0]
    ss_num=[0,0,0] #预测的二级结构类别的数目
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]):  # per aminoacid residue
            if real[i, j, 3] == 1:  # real[i, j, dataset.num_classes - 1] > 0 # if it is padding
                total = total - 1
            else:
                pred_ss=np.argmax(pred[i, j, 0:3])
                ss_num[pred_ss]+=1
                if real[i, j, pred_ss] > 0:
                    correct [pred_ss]+=1
    print(correct,ss_num,total)
    correct_rate=np.divide(correct,ss_num)
    return correct_rate
def q3_pred(y_true, y_pred):
    # q3=Q3_accuracy(y_true,y_pred)
    print(type(q3_pred))
    return K.mean(y_pred)


class Histories(keras.callbacks.Callback):
    def __init__(self,VALID=False):
        self.loss = []
        self.q3 = []
        self.acc=[]
        if not VALID:
            # self.X_test=get_pssm('test.npy')
            self.X_test=get_onehot('test.npy')
            self.Y_test=get_dssp('test.npy')
    def load_from_record(self,id):
        data=read_record(id)
        self.loss.append( data.loss)
        self.q3.append(data.q3)
        self.info=data.info
    def on_train_begin(self, logs={}):
        pass
        
#####  此处修改学习率变化函数
    def lr_change(self,epoch):
        if epoch ==20:
            K.set_value(self.model.optimizer.lr, .0001)
            print("lr changed")
        
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        if GRAD:
            self.lr_change(epoch)
        return
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(float(logs.get('loss')))
        self.acc.append(logs.get('acc'))
        if VALID:
            y_pred = self.model.predict(self.validation_data[0])
            y_real= self.validation_data[1]
        else:
            y_pred = self.model.predict(self.X_test)
            y_real= self.Y_test
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
def get_info(list,*args):
    list.append(args)


############
def test_saved():
    X_test=get_pssm('test.npy')
    Y_test=get_dssp('test.npy')
    m=load_model(MODEL_PATH)
    predictions = m.predict(X_test,)
    print("\n\nQ3 accuracy: E H C" + str(Q3_EHC(Y_test, predictions)) + "\n\n")
    p=dssp_trans(onehot=predictions)
    y=get_dssp_raw('test.npy')
    seq=get_seq('test.npy')
    print(seq[0])
    print(p[0])
    print(y[0])
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
def read_record(id):
    with open(file_name,'r',encoding='utf-8') as f:
        res=yaml.load(f.read())
        return res[id]
def add_record(id,record):
    data={}
    with open(file_name, "a",encoding='utf-8') as f:
        data[id]=record
        yaml.dump(data, f)
def rank(seq_source):
    seq=[]
    seq[:]=seq_source
    n=len(seq)
    for i in range(n) :
        for j in range(i,n) :
            if len(seq[i])>len(seq[j]):
                temp=seq[i]
                seq[i]=seq[j]
                seq[j]=temp
    return seq
def find_by_seq(seqs,seq):
    for i in range(len(seqs)):
        if seq==seqs[i]:
            return i
if __name__ == "__main__":
        print('测试')
        #画学习率
        # plot_record(5347,302,9969,6981,2966,info=['0.001','0.002','0.003','0.004','0.005'])
        #画rnn和cnn
        # plot_record(9969,4312,info=['rnn','cnn'])
        np.set_printoptions(threshold=np.inf)
        path=('train.npy')
        
        train=np.load(path).item()
        seq_source=train['seq']
        dssp=train['dssp']
        pssm=train['pssm']
        min=''
        mincount=800
        minid=0
        seq=rank(seq_source)
        
        print(find_by_seq(seq_source, seq[34]))
             
        
