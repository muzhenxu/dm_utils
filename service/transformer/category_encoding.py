from category_encoders import *
import numpy as np
from collections import Counter
import pandas as pd


class CountEncoder(object):
    """
    count encoding: Replace categorical variables with count in the train set.
    replace unseen variables with 1.
    Can use log-transform to be avoid to sensitive to outliers.
    Only provide log-transform with base e, because I think it's enough.


    Attributes
    ----------
    dmap: a collections.Counter(which like dict) map variable's values to its frequency.

    Example
    -------
    enc = countencoder()
    enc.fit(['a','b','c', 'b', 'c', 'c'])
    enc.transform(['a','c','b'])
    Out:
    array([ 0.        ,  1.09861229,  0.69314718])

    """

    def __init__(self, unseen_values=1, log_transform=True, smoothing=1):
        self.unseen_values = unseen_values
        self.log_transform = log_transform
        self.smoothing = smoothing
        self.map = {}

    def fit(self, X, y=None):
        """
        :param X: df
        :param y: None
        :return:
        """
        df = pd.DataFrame(X)

        for c in df.columns:
            dmap = Counter(X)
            for k in dmap.keys():
                dmap[k] += self.smoothing
            dmap['*******************unknown'] = self.unseen_values
            self.map[c] = dmap

        self.columns = df.columns

    def transform(self, X):
        """
        :param X: df
        :return:
        """
        # TODO: maybe use pd.Series with replace can faster. should test.
        df = pd.DataFrame(X)
        if df.columns != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        for c in self.columns:
            l = [i for i in df[c].unique() if i not in self.map[c].keys()]
            if len(l) > 0:
                df[c].replace(l, '*******************unknown', inplace=True)
            df[c] = df[c].replace(self.map[c])
        if self.log_transform:
            X = np.log(df)
        return X


def CategoryEncoder(method='CountEncoder', **kwargs):
    """
    contain these method: onehotencoder, labelencoder, targetencoder, countencoder(default)

    :param method: onehotencoder, labelencoder, targetencoder, countencoder(default)
    :return:
    """
    return eval(method)(**kwargs)
