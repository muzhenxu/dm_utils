import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn import metrics as mr
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import feature_encoding


def count_binary(a, event=1):
    """
    count the number of a's values
    :param a:
    :param event:
    :return:
    """
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count


def woe_iv(x, y, event=1, **kwargs):
    """
    calculate woe and information for a single feature
    :param x: 1-D numpy stands for single feature
    :param y: 1-D numpy array target variable
    :param event: value of binary stands for the event to predict
    :return: dictionary contains woe values for categories of this feature
             information value of this feature
    """
    x = np.array(x)
    y = np.array(y)

    enc = feature_encoding.binningencoder(**kwargs)
    enc.fit(x, y)
    x = enc.transform(x)

    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    woe_dict = {}
    iv = 0
    for x1 in x_labels:
        # this for array,如果传入的是pd.Series，y会按照index切片，但是np.where输出的却是自然顺序，会出错
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        if rate_event == 0:
            woe1 = -20
        elif rate_non_event == 0:
            woe1 = 20
        else:
            woe1 = math.log(rate_event / rate_non_event)
        woe_dict[x1] = woe1
        iv += (rate_event - rate_non_event) * woe1
    return woe_dict, iv


def iv_df(df, labels, columns=None, **kwargs):
    """
    compute iv for every column in columns with every label in labels
    :param df: dataframe
    :param labels: list
    :param columns: list, if None, all features except for labels will be computed iv with every label in labels.
    :return: dataframe
    """
    dic = defaultdict(dict)

    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    for t in labels:
        for c in columns:
            dic[t][c] = woe_iv(df[c], df[t], **kwargs)[
                1]
    df = pd.DataFrame(dic)
    df.columns = [['IV'] * df.shape[1], df.columns]
    return df


def ks_df(df, labels, columns=None):
    """
    compute ks for every column in columns with every label in labels using scipy.stats.ks_2samp
    :param df:
    :param labels: list
    :param columns:
    :return:
    """
    dic = defaultdict(dict)

    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    for t in labels:
        for c in columns:
            dic[t][c] = ks_2samp(df[df[t] == 1][c], df[df[t] == 0][c])[0]
    df = pd.DataFrame(dic)
    df.columns = [['KS'] * df.shape[1], df.columns]
    return df


def dt_auc(y, x, n_split=5, max_depth=5, min_samples_leaf=1, max_leaf_nodes=None, random_state=7):
    """
    compute auc for single feature with label by decision tree.
    :param y: array, target values
    :param x: array, single feature values
    :param n_split: int, the number of folds for cross validation.
    :param max_depth:
    :param min_samples_leaf:
    :param max_leaf_nodes:
    :return: float, auc
    """
    x, y = np.array(x).reshape(-1, 1), np.array(y)

    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                random_state=random_state)

    prob = []
    y_true = []
    skf = StratifiedKFold(n_split, shuffle=True)
    for tr_index, te_index in skf.split(x, y):
        dt.fit(x[tr_index], y[tr_index])
        prob.extend(list(dt.predict_proba(x[te_index])[:, 1]))
        y_true.extend(list(y[te_index]))
    auc1 = mr.roc_auc_score(y_true, prob)
    if auc1 < 0.5:
        auc1 = 1 - auc1

    # 对于单变量，尤其是分数类型的单变量，直接计算auc会更加准确，因为通过dt去训练会丢失信息
    auc2 = mr.roc_auc_score(y, x)
    if auc2 < 0.5:
        auc2 = 1 - auc2
    if auc1 < auc2:
        auc = auc2
    else:
        auc = auc1
    return auc


def auc_df(df, labels, columns=None, n_split=5, max_depth=5, min_samples_leaf=1, max_leaf_nodes=None, random_state=7):
    """
    compute auc for every column in columns with every label in labels
    :param df:
    :param labels: list
    :param columns:
    :return:
    """
    dic = defaultdict(dict)

    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    for t in labels:
        for c in columns:
            auc = dt_auc(df[t], df[c], n_split=n_split, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         max_leaf_nodes=max_leaf_nodes, random_state=random_state)
            if auc < 0.5:
                auc = 1 - auc
            dic[t][c] = auc

    df = pd.DataFrame(dic)
    df.columns = [['AUC'] * df.shape[1], df.columns]
    return df


def feature_evaluation_single(df, labels, columns=None, **kwargs):
    """
    compute auc, iv, ks for every column in columns with every label in labels
    :param df:
    :param labels: list
    :param columns:
    :return:
    """
    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    df_ks = ks_df(df, labels, columns)
    df_auc = auc_df(df, labels, columns)
    df_iv = iv_df(df, labels, columns, **kwargs)
    df = pd.concat([df_auc, df_iv, df_ks], axis=1)
    df = df.sort_index(axis=1)
    return df


def feature_evaluation(df, labels, hue=None, columns=None, **kwargs):
    """
    compute auc, iv, ks for every column in columns with every label in labels group by hue.
    :param df:
    :param labels: list
    :param columns:
    :param hue: str
    :return:
    """
    if columns is None:
        columns = [c for c in df.columns if c not in labels and c != hue]

    res = pd.DataFrame()
    if hue is None:
        return feature_evaluation_single(df, labels, columns=columns, **kwargs)
    s = sorted(df[hue].unique())
    for i in s:
        temp = feature_evaluation_single(df[df[hue] == i], labels, columns=columns, **kwargs)
        temp.columns = pd.MultiIndex.from_product([temp.columns.levels[0], temp.columns.levels[1], [i]])
        res = pd.concat([res, temp], axis=1)
    res = res.sort_index(axis=1)
    return res

if __name__ == '__main__':
    try:
        df = pd.read_pickle('utils/iv_test.pkl')
    except:
        df = pd.read_pickle('iv_test.pkl')
    t1 = feature_evaluation(df, labels=['y'], max_depth=3)
    t2 = feature_evaluation(df, labels=['y'], max_depth=4)
    t3 = feature_evaluation(df, labels=['y'], max_depth=5)
    t4 = feature_evaluation(df, labels=['y'], binning_method='qcut', bins=5)
    t5 = feature_evaluation(df, labels=['y'], binning_method='qcut', bins=10)
    t6 = feature_evaluation(df, labels=['y'], binning_method='qcut', bins=15)
    t7 = feature_evaluation(df, labels=['y'], binning_method='cut', bins=5)
    t8 = feature_evaluation(df, labels=['y'], binning_method='cut', bins=10)
    t9 = feature_evaluation(df, labels=['y'], binning_method='cut', bins=15)
    print(t1, t2, t3, t4, t5, t6, t7, t8, t9)