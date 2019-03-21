'''
2.0版本
#区分末端氨基酸
增加空白氨基酸、空白二级结构表示
增加双向神经网络支持
'''
import tensorflow as tf
from tensorflow.contrib import rnn 
from tensorflow.nn import rnn_cell
import numpy as np
import time
##
n_input=21
n_window=13
n_hidden=256
n_class=4
learning_rate=0.001
train_iters=260000
batch_size=64
display_step=100
class DataSet:
    def __init__(self,path,n_window):
        self.aa_n=0
        self.seq_n=0
        self.path=path
        self.WINDOW_SIZE=n_window
        ds=np.load(path).item()
        self.seq=np.array(ds['seq'])
        self.SEQ_NUM=len(self.seq)
        self.pssm=np.array(ds['pssm'])
        self.dssp=np.array(ds['dssp'])
        self.dssp_onehot()
        self.padding()
        self.total_aa=len(self.pssm)
        self.index_batch=0
       
        print(self.total_aa)
        
        # assert 1==2
        del ds
    def padding(self):
        pssm_none=np.array([0]*20+[1])
        dssp_none=np.array([0,0,0,1])
        half_window=self.WINDOW_SIZE//2
        pssm_tmp=np.array([pssm_none]*half_window)
        dssp_tmp=np.array([dssp_none]*half_window)
        for i in range(self.SEQ_NUM):
            ret=np.zeros((len(self.pssm[i]),21))
            ret[:,0:20]=self.pssm[i][:,:]
            self.pssm[i]=np.concatenate( [pssm_tmp,ret,pssm_tmp])
            self.dssp[i]=np.concatenate([dssp_tmp,self.dssp[i],dssp_tmp])
        


    def all_test(self):
        X=[]
        Y=[]
        tmp=np.vstack(self.pssm)
        tmpy=np.vstack(self.dssp)
        for i in range(len(tmp)-self.WINDOW_SIZE):
            X.append(tmp[i:i+self.WINDOW_SIZE])
            Y.append(tmpy[i+self.WINDOW_SIZE//2])
        return X,Y
    def next_batch(self,batch_size):
        X=[]
        Y=[]
        half_window=self.WINDOW_SIZE//2
        seq_remain=len(self.dssp[self.seq_n])-self.aa_n
        if seq_remain<self.WINDOW_SIZE:
            self.aa_n=0
            self.seq_n=(self.seq_n+1)%self.SEQ_NUM
            seq_remain=len(self.dssp[self.seq_n])
        output_size=np.min([seq_remain-self.WINDOW_SIZE+1,batch_size])
        
        # if self.aa_n==384:
        #         print(len(self.pssm[self.seq_n]))
        for i in range(output_size):
            start=self.aa_n+i
            end=self.aa_n+self.WINDOW_SIZE+i
            mid=self.aa_n+half_window+i
            # if self.aa_n==384:
            #     print(start,mid,end)
            X.append(self.pssm[self.seq_n][start:end][:])
            Y.append(self.dssp[self.seq_n][mid])
        X=np.array(X)
        Y=np.array(Y)
       
                
        # print(X.shape,Y.shape,self.seq_n,self.aa_n,output_size)
        self.aa_n+=batch_size
        return X,Y
    def dssp_onehot(self):
        num = len(self.dssp)
        ret=[]
        onehot_dict = {'E': 0, 'H': 1, '-': 2,' ':3}
        for i in range(num):
            ret.append([])
            for j in self.dssp[i]:
                one=[0,0,0,0]
                k = onehot_dict[j]
                one[k]=1
                ret[i].append(one)
        self.dssp=ret

# np.random.seed(100)
def RNN(x,weights,biases):
    x=tf.transpose(x,[1,0,2]) #64,13,21 ->13,64,21
    x=tf.reshape(x,[-1,n_input]) #13*64,21
    #将x 切成n_steps=28个子张量
    x=tf.split(axis=0,num_or_size_splits=n_window,value=x)
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
def test(epochs=100):
    time1=time.time()
    Train=DataSet('train.npy',n_window)
    x=tf.placeholder('float',[None,n_window,n_input])
    y=tf.placeholder('float',[None,n_class])
    weights={ 'out':tf.Variable(tf.random_normal([2*n_hidden,n_class]))
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
        time2=time.time()
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
        time3=time.time()
        train_time=int(time3-time2)
        Test=DataSet('test.npy',n_window)
        test_len=Test.total_aa-100
        test_x,test_y=Test.all_test()
        print("隐藏层:%d,迭代：%d次,时间:%d秒"\
            %(n_hidden,train_iters,train_time))
        print("准确率:",\
            sess.run(accuracy,feed_dict={x:test_x,y:test_y}))
if __name__ == "__main__":
    test(50)

'''
隐藏层:256,迭代：100000次,时间:12秒
准确率: 0.6471092
隐藏层:256,迭代：150000次,时间:19秒
准确率: 0.6739973
n_hidden:256,迭代：200000,时间:25
准确率: 0.70055896
隐藏层:256,迭代：250000次,时间:31秒
准确率: 0.71430904
隐藏层:256,迭代：260000次,时间:32秒
准确率: 0.6786079
隐藏层:256,迭代：270000次,时间:34秒
准确率: 0.71210575
n_hidden:256,迭代：300000,时间:38
准确率: 0.65988004

'''