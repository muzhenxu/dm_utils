import numpy as np
import xlearn as xl
from .df2libffm import FfmEncoder

class xl_model(object):
    def __init__(self, cat='lr', task='binary', init=0.1, epoch=10, lr=0.1, k=4, reg_lambda=1.0, opt='sgd', metric='auc'):
        self.task = task
        self.init = init
        self.epoch = epoch
        self.lr = lr
        self.k = k
        self.reg_lambda = reg_lambda
        self.opt = opt
        self.metric = metric
        self.cat = cat

    def fit(self, df, label, path='datasource/train.ffm'):
        if self.cat == 'lr':
            self.clf = xl.LRModel()
        elif self.cat == 'fm':
            self.clf = xl.FMModel()
        elif self.cat == 'ffm':
            self.clf = xl.FFMModel()
        else:
            raise ValueError(self.cat, ' is an invalid value for param cat.')

        self.fe = FfmEncoder(df)
        self.fe.fit(df, label, path=path)

        self.clf.fit()


