import yaml

file_name='a.yaml'
def read_record(id):
    with open(file_name,'r',encoding='utf-8') as f:
        res=yaml.load(f.read())
        return res[id]
def add_record(id,record):
    data={}
    with open(file_name, "a",encoding='utf-8') as f:
        data[id]=record
        yaml.dump(data, f)
if __name__ == "__main__":
    record={'info':'lr 0.009','q3':[22,33,44,55],'loss':[332,44,44,55]}
    add_record(2,record)
    read_record(2)

'''
157 学习率0.005 40周期
801 学习率 0.007 40周期
'''