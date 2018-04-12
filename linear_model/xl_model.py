import numpy as np
import xlearn as xl
from .df2libffm import FFMEncoder
import pandas as pd


class xl_model(object):
    def __init__(self, model_type='lr', task='binary', init=0.1, epoch=10, lr=0.1, k=4, reg_lambda=1.0, opt='sgd',
                 metric='auc', cutoff=0):
        self.task = task
        self.init = init
        self.epoch = epoch
        self.lr = lr
        self.k = k
        self.reg_lambda = reg_lambda
        self.opt = opt
        self.metric = metric
        self.model_type = model_type
        self.cutoff = cutoff
        self.params = {'task': task, 'init': init, 'epoch': epoch, 'lr': lr, 'k': k,
                       'reg_lambda': reg_lambda, 'opt': opt, 'metric': metric}

    def fit(self, df, label, eva_df=None, eva_label=None, path='datasource/train.ffm', eva_path='datasource/valid.ffm',
            model_path='datasource/ffm_model.out'):
        if (eva_df is None) ^ (eva_label is None):
            raise Exception('params eva_df, eva_df must be all None or all have value.')

        if self.model_type == 'lr':
            self.clf = xl.create_ffm()
        elif self.model_type == 'fm':
            self.clf = xl.create_fm()
        elif self.model_type == 'ffm':
            self.clf = xl.create_linear()
        else:
            raise ValueError(self.model_type, ' is an invalid value for param cat.')

        self.fe = FFMEncoder(df)
        self.fe.fit(df, self.cutoff)
        self.transform(df, label, path)
        if eva_df is not None:
            self.transform(eva_df, eva_label, eva_path)

        self.clf.setTrain(path)
        if eva_df is not None:
            self.clf.setlidate(eva_path)

        self.clf.fit(self.params, model_path)
        self.model_path = model_path

    def predict(self, df, path='datasource/test.ffm', out_path='datasource/pred.txt'):
        self.fe.transform(df, path)
        self.clf.setTest(path)
        self.clf.setSigmoid()
        self.clf.predict(self.model_path, out_path)
        pred = pd.read_csv(out_path)
        return pred
