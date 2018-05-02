"""
Use this class to process categorical variables.
document: https://www.slideshare.net/HJvanVeen/feature-engineering-72376750
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import bisect
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import math


def dt_cut_points(x, y, max_depth=4, min_samples_leaf=0.05, max_leaf_nodes=None, random_state=7):
    """
    A decision tree method to bin continuous variable to categorical one.
    :param x: The training input samples
    :param y: The target values
    :param max_depth: The maximum depth of the tree
    :param min_samples_leaf: int, float, The minimum number of samples required to be at a leaf node
    :param max_leaf_nodes: Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    :return: The list of cut points
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                random_state=random_state)
    dt.fit(np.array(x).reshape(-1, 1), np.array(y))
    th = dt.tree_.threshold
    f = dt.tree_.feature

    # 对于没有参与分裂的节点，dt默认会给-2,所以这里要根据dt.tree_.feature把-2踢掉
    return sorted(th[np.where(f != -2)])


def get_cut_points(X, y=None, bins=10, binning_method='dt', precision=8, **kwargs):
    if binning_method == 'cut':
        _, cut_points = pd.cut(X, bins=bins, retbins=True, precision=precision)
    elif binning_method == 'qcut':
        _, cut_points = pd.qcut(X, q=bins, retbins=True, duplicates='drop', precision=precision)
    elif binning_method == 'dt':
        cut_points = dt_cut_points(X, y, **kwargs)
    else:
        raise ValueError("binning_method: '%s' is not defined." % binning_method)

    if binning_method != 'dt':
        cut_points = cut_points[1:-1]

    cut_points = list(cut_points)
    cut_points.append(np.inf)
    cut_points.insert(0, -np.inf)

    return cut_points


def woe(x, y, woe_min=-20, woe_max=20):
    # TODO: woe_min&woe_max设置是否合理？
    """

    :param x: array
    :param y: array
    :return:
    """
    x = np.array(x)
    y = np.array(y)

    pos = (y == 1).sum()
    neg = (y == 0).sum()

    dmap = {}

    for k in np.unique(x):
        indice = np.where(x == k)
        pos_r = (y[indice] == 1).sum() / pos
        neg_r = (y[indice] == 0).sum() / neg

        if pos_r == 0:
            woe1 = woe_min
        elif neg_r == 0:
            woe1 = woe_max
        else:
            woe1 = math.log(pos_r / neg_r)

        dmap[k] = woe1

    return dmap


class WOEEncoder():
    def __init__(self, bins=10, binning_method='dt', woe_min=-20, woe_max=20, nan_thr=0.01, **kwargs):
        self.bins = bins
        self.binning_method = binning_method
        self.kwargs = kwargs
        self.woe_min = woe_min
        self.woe_max = woe_max
        self.nan_thr = nan_thr
        self.map = {}

    def fit(self, X, y):
        df = pd.DataFrame(X)
        y = pd.DataFrame(y)
        self.columns = df.columns

        label = 'label'
        y.columns = [label]
        for c in df.columns:
            tmp = pd.concat([df[c], y], axis=1)

            nan_woe = self.nan_woe_cmpt(df, c, label, self.nan_thr, self.woe_min, self.woe_max)

            tmp = tmp.dropna()

            cut_points = get_cut_points(tmp[c], tmp[label], self.bins, self.binning_method, **self.kwargs)
            dmap = woe(tmp[c], tmp[label], self.woe_min, self.woe_max)
            dmap[np.nan] = nan_woe

            self.map[c] = {'cut_points': cut_points, 'dmap': dmap}

    def transform(self, X):
        df = pd.DataFrame(X)
        if df.columns != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        for c in self.columns:
            df[c] = pd.cut(df[c], bins=self.map[c]['cut_points'])
            df[c] = df[c].replace(self.map[c]['dmap'])
        return df

    @staticmethod
    def nan_woe_cmpt(df, col, label='label', nan_thr=0.01, woe_min=-20, woe_max=20):
        m = df[label].mean()
        pos_base = int(df.shape[0] * m * nan_thr)
        neg_base = int(df.shape[0] * (1 - m) * nan_thr)
        tmp = df[df[col].isnull()]
        t = tmp[label].shape[0]
        pos = tmp[label].sum()
        neg = t - pos
        pos_r = pos + pos_base
        neg_r = neg + neg_base

        if pos_r == 0:
            woe = woe_min
        elif neg_r == 0:
            woe = woe_max
        else:
            woe = math.log(pos_r / neg_r)

        return woe


class NothingEncoder():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return X


def ContinousEncoder(method='NothingEncoder', **kwargs):
    """
    contain these method: onehotencoder, labelencoder, targetencoder, countencoder(default)

    :param method: onehotencoder, labelencoder, targetencoder, countencoder(default)
    :return:
    """
    return eval(method)(**kwargs)
