# graduation
毕业设计 蛋白质二级结构预测  
联系方式 qq：527474091
# 使用教程 
## 操作系统
linux、window都可以，mac没设备测试(T_T)  
## 最新测试环境:  
 ubuntu 20.04  
 python 3.8  
 tensorflow '2.2.0-rc2'
## 搭建运行所需环境
1. 如果你的机器有nvidia显卡，并支持cuda,则可以大大加快训练的速度 
具体安装步骤请参考 https://tensorflow.google.cn/install  
这里只给出cpu版本的环境搭建方法

```shell
# 安装pip3，以便安装python包
sudo apt install python3-pip  
# 使用清华pip镜像，下载更快
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  
# 安装运行所需的扩展包
pip3 install numpy keras matplotlib tensorflow  --user
```
## 运行方法
>python3 main.py  

如果需要修改运行效果，请查看main.py
# 详细说明
- train.npy为训练集，test.npy为测试集
- main.py为主程序，运行它，就能得到我预设的效果，修改它就可以得到更多的功能
- mytools.py 为我写的工具箱，方便程序编写
- saved_model.h5 保存训练后的模型，方便多次训练
- a.yaml 保存了训练中 误差率 loss和准确率q3 随训练代数epoch的变化
- 1.mp3  为训练完成的通知铃声
