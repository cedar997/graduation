from flask import Flask, request, render_template, url_for, jsonify, abort
from main import *
import time

app = Flask(__name__)
#全局训练结果
his=None
#全局的模型
model=None
@app.route('/')
def hello_world():
    #return "hllo"
    return render_template('index.html')
@app.route('/run',methods=['POST'])
def run_fun():
    global his,model
    his = Histories()
    jsons=request.json
    print(jsons['net'])
    para=Para()
    para.lr=float(jsons['lr'])
    para.net=jsons['net']
    para.epoch=int(jsons['epoch'])
    #获取模型
    model = net.getNetByPara(para)
    train_mode_get_his(model,his,para)
    ret={
        'q3': his.get_q3()
    }
    return jsonify(ret)



#获取训练进度
@app.route('/index/<int:index>',methods=['GET'])
def get_index(index):
    global his
    if his:
        time_limit=100
        # 自旋，等待训练完成
        while (time_limit>0):
            time.sleep(1)
            if(his.get_index()>=index):
                break
            time_limit-=10

        ret = {
            'index': his.get_index(),
            'q3': his.get_q3(),
            'acc': his.get_acc(),
            'loss': his.get_loss()
        }
        return jsonify(ret)
    ret={
        'index': 0,
        'q3': []
    }
    return jsonify(ret)
@app.route('/data',methods=['GET'])
def load_data():
    from models.net import net_map
    return jsonify(net_map)

if __name__ == '__main__':
    
    app.run(host='localhost',debug=True)
