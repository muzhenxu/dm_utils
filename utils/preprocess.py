import pandas as pd
import numpy as np
from collections import Counter
from utils.mdlp import MDLP
from utils.woe import WOE
from collections import defaultdict
from sklearn import metrics as mr
import xgboost as xgb
from sklearn.model_selection import train_test_split

# def del_redundance_cols(df_origin, nan_threshold=100, same_threshold=20, drop_duplicates=False, silent=False):
#     """
#     剔除完全缺失字段，完全同值字段
#     删除重复case
#     :param df_origin:
#     :param silent:
#     :param threshold:
#     :return:
#     """
#     df = df_origin.copy()
#
#     df_nan = pd.DataFrame(df.isnull().sum(axis=0), columns=['num'])
#     df_nan['notnull_num'] = df.shape[0] - df_nan['num']
#     df_nan['percent'] = df_nan['num'] / df.shape[0]
#     df_nan = df_nan.sort_values(by=['num'], ascending=False)
#
#     nunique_value = df.apply(lambda c: c.nunique())
#     df_nan['only_value'] = nunique_value
#
#     same_value = df.apply(lambda c: c.value_counts().iloc[0])
#     df_nan['same_value'] = same_value
#     df_nan['same_ratio'] = same_value / df.shape[0]
#
#     # drop_cols = nunique_value.index[nunique_value < 2]
#
#     # 计算缺失值占比过多的columns
#     nan_ratio = 1 - nan_threshold / df.shape[0]
#     nan_cols = df_nan.index[df_nan.percent > nan_ratio]
#
#     # 计算同一值过多的columns
#     same_ratio = 1 - same_threshold / df.shape[0]
#     same_cols = df_nan.index[df_nan.same_ratio > same_ratio]
#
#     df = df.drop(same_cols, axis=1)
#     df = df.drop(nan_cols, axis=1)
#
#     n = df.shape[0]
#     if drop_duplicates:
#         df = df.drop_duplicates()
#     m = df.shape[0]
#
#     if not silent:
#         print(df_nan)
#         print('columns which all values are nan: ', nunique_value.index[nunique_value == 0])
#         print('columns which all values are the same: ', nunique_value.index[nunique_value == 1])
#         print('columns which notnull values\' num below %s and nan ratio beyond %s: ' % (nan_threshold, nan_ratio),
#               nan_cols.values)
#         print('columns which not same values\' num below %s and same ratio beyond %s: ' % (same_threshold, same_ratio),
#               same_cols.values)
#         print('%s cases is duplicates.' % (n - m))
#
#     return df


def del_redundance_cols(df_origin, nan_threshold=100, same_threshold=20, drop_duplicates=False, silent=False):
    """
    剔除完全缺失字段，完全同值字段
    删除重复case
    :param df_origin:
    :param silent:
    :param threshold:
    :return:
    """
    df = df_origin.copy()

    df_nan = pd.DataFrame(df.isnull().sum(axis=0), columns=['num'])
    df_nan['notnull_num'] = df.shape[0] - df_nan['num']
    df_nan['percent'] = df_nan['num'] / df.shape[0]
    df_nan = df_nan.sort_values(by=['num'], ascending=False)

    nunique_value = df.apply(lambda c: c.nunique())
    df_nan['only_value'] = nunique_value

    same_value = df.apply(lambda c: c.value_counts().iloc[0])
    df_nan['same_value'] = same_value
    df_nan['same_ratio'] = same_value / df.shape[0]

    # drop_cols = nunique_value.index[nunique_value < 2]

    # 计算缺失值占比过多的columns
    nan_ratio = 1 - nan_threshold / df.shape[0]
    nan_cols = df_nan.index[df_nan.percent > nan_ratio]

    # 计算同一值过多的columns
    same_ratio = 1 - same_threshold / df.shape[0]
    same_cols = df_nan.index[df_nan.same_ratio > same_ratio]

    df = df.drop(same_cols, axis=1)
    df = df.drop(nan_cols, axis=1)

    n = df.shape[0]
    if drop_duplicates:
        df = df.drop_duplicates()
    m = df.shape[0]

    if not silent:
        print(df_nan)
        print('columns which all values are nan: ', nunique_value.index[nunique_value == 0])
        print('columns which all values are the same: ', nunique_value.index[nunique_value == 1])
        print('columns which notnull values\' num below %s and nan ratio beyond %s: ' % (nan_threshold, nan_ratio),
              nan_cols.values)
        print('columns which not same values\' num below %s and same ratio beyond %s: ' % (same_threshold, same_ratio),
              same_cols.values)
        print('%s cases is duplicates.' % (n - m))

    return df

def cmp_array(a_origin, b_origin):
    """
    比较两个数组是否完全相等，此处暂时不考虑含nan情况,不然结果有误
    :param a_origin:
    :param b_origin:
    :return:
    """
    a = np.array(a_origin.copy())
    b = np.array(b_origin.copy())
    if (a == 0).all():
        if Counter(b) == 1:
            return True
        else:
            return False
    for k, v in enumerate(a):
        if v != 0:
            ratio = b[k] / a[k]
            break

    a = a * ratio
    return ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def del_duplicate_cols(df_origin, del_cols=True):
    """
    删除重复字段,同比例字段
    refer: https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns/46955471#46955471?newreg=1dd48de235cf44eca5fd0aa59c27f2cc
    :param df_origin:
    :param del_cols:
    :return:
    """
    df = df_origin.copy()
    groups = df.columns.to_series().groupby(df.dtypes).groups
    dups = []
    for t, v in groups.items():
        cs = df[v].columns
        vs = df[v]
        lcs = len(cs)
        for i in range(lcs):
            dup = [cs[i]]
            if cs[i] in sum(dups, []):
                continue
            ia = vs.iloc[:, i].values
            for j in range(i + 1, lcs):
                if cs[j] in sum(dups, []):
                    pass
                ja = vs.iloc[:, j].values
                if cmp_array(ia, ja):
                    dup.append(cs[j])
            if len(dup) > 1:
                dups.append(dup)
    print('duplicate columns: ', dups)

    if del_cols:
        dups_drop = sum([l[:-1] for l in dups], [])
        df = df.drop(dups_drop, axis=1)

    return df


def woe_translate(df_origin, labels, columns=None, onehot=True, del_origin_columns=False):
    """
    woe encoder
    :param df_origin:
    :param columns:
    :param labels: type: list
    :return:
    """
    df = df_origin.copy()
    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    woe = WOE()
    mdlp = MDLP()

    dic_cut_points = {}
    dic_woe = {}
    dic_iv = defaultdict(dict)
    dic_nmi = defaultdict(dict)

    for c in columns:
        if df[c].nunique() <= 1:
            continue
        for t in labels:
            dic_cut_points[c] = mdlp.cut_points(np.array(df[c]), np.array(df[t]))
            df['%s_%s_mdlp' % (c, t)] = mdlp.discretize_feature(df[c], dic_cut_points[c])
            dic_woe[c], dic_iv[t][c] = woe.woe_single_x(df['%s_%s_mdlp' % (c, t)], df[t])
            dic_nmi[t][c] = mr.normalized_mutual_info_score(df['%s_%s_mdlp' % (c, t)], df[t])
            df['%s_%s_woe' % (c, t)] = df['%s_%s_mdlp' % (c, t)].replace(dic_woe[c].keys(), dic_woe[c].values())
            if onehot:
                df = pd.concat([df, pd.get_dummies(df['%s_%s_mdlp' % (c, t)], prefix='%s_%s_mdlp' % (c, t))], axis=1)

            if del_origin_columns:
                del df['%s_%s_mdlp' % (c, t)]
            print(c, t)

    if del_origin_columns:
        df = df.drop(columns, axis=1)

    df_iv = pd.DataFrame(dic_iv)
    df_nmi = pd.DataFrame(dic_nmi)
    df_iv.columns = [['IV'] * df_iv.shape[1], df_iv.columns]
    df_nmi.columns = [['NMI'] * df_nmi.shape[1], df_nmi.columns]
    df_iv_nmi = pd.concat([df_iv, df_nmi], axis=1)

    return df, df_iv_nmi, dic_cut_points, dic_woe

def onehot(df_origin, del_origin_columns=True):
    df = df_origin.copy()
    cols = df.columns

    for c in cols:
        if (df[c].nunique() <= 2) | (df[c].nunique() > 20):
            continue
        df = pd.concat([df, pd.get_dummies(df['%s' % c], prefix='%s' % c)], axis=1)
        if del_origin_columns:
            del df[c]
    return df


