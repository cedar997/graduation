from sqlalchemy import create_engine,ForeignKey
from sqlalchemy.orm import sessionmaker,relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer,Float, Text,DATE,func
import sqlalchemy.types as types
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from itsdangerous import BadSignature, SignatureExpired


Base = declarative_base()
def to_dict(self):
    return {c.name: getattr(self, c.name, None) for c in self.__table__.columns}
#指定基类的to_dict后，子类不用写
Base.to_dict = to_dict
class Para(Base):
    __tablename__ = 't_para'
    id=Column(Integer(),primary_key=True)
    lr=Column(Float())
    batch=Column(Integer())
    net=Column(String())
    epoch=Column(Integer())
class ScoreBase(Base):
    __tablename__ = 't_score'
    sid=Column(Integer(),primary_key=True)
    cmt_count=Column(Integer())
    score = Column(Integer())
    scenery = Column(Integer())
    traffic = Column(Integer())
    price = Column(Integer())
    food = Column(Integer())


class MyDb(object):
    def __init__(self):
        self.engine =create_engine('mysql+mysqlconnector://root:cedar@localhost:3306/test?charset=utf8mb4&auth_plugin=mysql_native_password')
        DBSession = sessionmaker(bind=self.engine)
        self.session = DBSession()
        self.SECRET_KEY = 'emtyyds'