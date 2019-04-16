import abc
from abc import abstractmethod

class Evaluation:
    __metaclass__ = abc.ABCMeta
    @classmethod
    def __init__(self):
        pass


    @abstractmethod
    def compute(self,train_label,label_predict,label_predict_proba):
        pass

