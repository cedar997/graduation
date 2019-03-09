import numpy as np
import math 
def load_file(path='train.npy'):
    #训练集#训练数量1348
    train=np.load(path).item()
    seq=train['seq']
    #PSSM位置特定评分矩阵20N
    pssm=train['pssm'] 
    dssp=train['dssp']
    #测试集#测试数量149
    return seq,pssm,dssp
def create_window_data(final_pssm, win_size):
    window_data = []
    j = win_size
    
    padded_table = np.pad(final_pssm, ((0,0),(math.floor(win_size/2),math.floor(win_size/2))), 
                        mode='constant', constant_values=0)

    for i in range(len(padded_table[0]) - (win_size-1)):
        extracted_table = padded_table[:,i:j]
        j += 1
        window_data.append(extracted_table)
    
    return np.array(window_data)
    
def format_pssm(train_pssm):
    ##连成一起
    pssm_final=[]
    for i in range(len(train_pssm)):
        for j in train_pssm[i]:
            pssm_final.append(j)
    ##创建窗口
    pssm_final= np.transpose(np.vstack(pssm_final))
    windows=create_window_data(pssm_final,21)
    ####
    X = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 1)
    return X
###
def format_dssp(train_dssp):
    output_data = []
    for i in range(len(train_dssp)):
            for j in range(len(train_dssp[i])):
                if train_dssp[i][j]=='E':
                    output_data.append([1,0,0])
                elif train_dssp[i][j]=='H':
                    output_data.append([0,1,0])
                else:
                    output_data.append([0,0,1])
    Y=np.array (output_data)
    return Y
###建模
