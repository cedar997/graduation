import tensorflow as tf
from tensorflow.contrib import rnn 
from tensorflow.nn import rnn_cell
import numpy as np
##
n_input=20
n_window=13
n_hidden=128
n_class=3
learning_rate=0.001
train_iters=500000
batch_size=64
display_step=10
SEQ_SIZE_MAX = 760
class DataSet:
    def __init__(self,path,window_size):
        self.path=path
        self.window_size=window_size
        ds=np.load(path).item()
        self.seq=np.array(ds['seq'])
        self.pssm=np.array(ds['pssm'])
        self.dssp=np.array(ds['dssp'])
        self.flaten()
        self.total_aa=len(self.pssm)
        self.index_batch=0
        del ds
    def flaten(self):
        pssm=np.empty([0,20])
        for item in self.pssm:
            pssm=np.append(pssm,item,axis=0)
        self.pssm=pssm
        print(len(pssm))
        self.dssp=self.get_flaten_dssp()
    def next_batch(self,batch_size):
        n=self.index_batch
        self.index_batch+=batch_size
        if self.index_batch>=self.total_aa-batch_size:
            self.index_batch=0
            print('all taked ')
        X=[]
        Y=[]
        for i in range(batch_size):
            
            X.append(self.pssm[n+i:n+i+self.window_size])
            Y.append(self.dssp[n+i+self.window_size//2])
        return X,Y
    def get_flaten_dssp(self):
        num = len(self.dssp)
        # ret = np.zeros((num, SEQ_SIZE_MAX, 4))
        ret=[]
        onehot_dict = {'E': 0, 'H': 1, '-': 2,}
        for i in range(num):
                for j in range(len(self.dssp[i])):
                    one=[0,0,0]
                    k = onehot_dict[self.dssp[i][j]]
                    one[k]=1
                    ret.append(one)
           
        return ret

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
def RNN(x,weights,biases):
    x=tf.transpose(x,[1,0,2]) #64,13,21 ->13,64,21
    x=tf.reshape(x,[-1,n_input]) #13*64,21
    #将x 切成n_steps=28个子张量
    x=tf.split(axis=0,num_or_size_splits=n_window,value=x)
    lstm_cell=rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
    outputs,states=rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['out'])+biases['out']
def test(epochs=100):
    Train=DataSet('train.npy',n_window)
    x=tf.placeholder('float',[None,n_window,n_input])
    y=tf.placeholder('float',[None,n_class])
    weights={ 'out':tf.Variable(tf.random_normal([n_hidden,n_class]))
    }
    biases={
            'out':tf.Variable(tf.random_normal([n_class]))
    }
    pred=RNN(x,weights,biases)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=pred,labels=y
    ))
    optimizer=tf.train.AdamOptimizer\
        (learning_rate=learning_rate).minimize(cost) 
    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step=1
        while step*batch_size<train_iters:
            batch_x,batch_y=Train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
            if step % display_step==0 :
                acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
                loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +\
    "{:.6f}".format(loss) + ", Training Accuracy= " +\
    "{:.5f}".format(acc))
            step+=1
        print("optimization finished")
        Test=DataSet('test.npy',n_window)
        test_len=Test.total_aa-100
        test_x,test_y=Test.next_batch(test_len)
        print("test accuracy:",\
            sess.run(accuracy,feed_dict={x:test_x,y:test_y}))
if __name__ == "__main__":
    test(50)