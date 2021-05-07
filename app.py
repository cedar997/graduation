from flask import Flask, request, render_template, url_for, jsonify, abort
from main import *


app = Flask(__name__)


@app.route('/')
def hello_world():
    #return "hllo"
    return render_template('index.html')
@app.route('/run',methods=['POST'])
def run_fun():
    net=request.json
    print(net['net'])
    para=Para()
    para.lr=float(net['lr'])
    para.net=net['net']
    para.epoch=int(net['epoch'])
    his=test(para)
    return jsonify(his.getq3())

@app.route('/data',methods=['GET'])
def load_data():
    from models.net import net_map
    return jsonify(net_map)

if __name__ == '__main__':
    
    app.run(host='0.0.0.0',debug=True)
