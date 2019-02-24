import numpy as np
#训练集#训练数量1348
train=np.load('train.npy').item()
train_seq=train['seq']
train_pssm=train['pssm']
#dssp只含有EH-,数量跟seq相等
train_dssp=train['dssp']
#测试集#测试数量149
test=np.load('test.npy').item()
test_seq=test['seq']
test_pssm=test['pssm']
test_dssp=test['dssp']

