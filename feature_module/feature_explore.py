import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
import os


def desc_df(df_origin):
    df = df_origin.copy()

    df_desc = pd.DataFrame(df.isnull().sum(axis=0), columns=['null_num'])
    df_desc['notnull_num'] = df.shape[0] - df_desc['null_num']
    df_desc['notnull_ratio'] = df_desc['notnull_num'] / df.shape[0]

    nunique_value = df.apply(lambda c: c.nunique())
    df_desc['diff_values_num'] = nunique_value

    try:
        same_value = df.apply(lambda c: c.value_counts().iloc[0])
    except:
        same_value = 0
    df_desc['most_value_num'] = same_value
    df_desc['same_ratio'] = same_value / df.shape[0]

    return df_desc


def cover_stats(df, path=None, feature_map_dict=None):
    data_fg_r = (df.notnull().sum(axis=1) > 0).sum() / df.shape[0]

    t = desc_df(df)
    t = t[['notnull_ratio']]
    t.columns = ['覆盖率']

    t['饱和度'] = t['覆盖率'] / data_fg_r
    if feature_map_dict:
        t.index = pd.MultiIndex.from_tuples([(i, feature_map_dict[i]) for i in t.index])
        t1 = pd.DataFrame(columns=['覆盖率', '饱和度'])
        t1.ix[('dataset', '全数据集'),] = [data_fg_r, np.nan]
        t = pd.concat([t1, t], axis=0)
    else:
        t1 = pd.DataFrame(columns=['覆盖率', '饱和度'])
        t1.loc['全数据集'] = [data_fg_r, np.nan]
        t = pd.concat([t1, t], axis=0)

    if not path:
        if not os.path.exists('reportsource'):
            os.mkdir('reportsource')
    t.to_pickle('reportsource/覆盖率统计.pkl')
    return t


def target_stats(df_origin, target_cols, path=None):
    df = df_origin.copy()
    df['是否匹配上'] = (df.notnull().sum(axis=1) > len(target_cols)).astype(int).replace([1, 0], ['是', '否'])

    t = pd.pivot_table(df, index='是否匹配上', values=target_cols, margins=True)

    if not path:
        if not os.path.exists('reportsource'):
            os.mkdir('reportsource')

    t.to_pickle('reportsource/逾期率统计.pkl')
    return t


class data_preprocess(object):
    def __init__(self, notnull_threshold=100, notsame_threshold=20, drop_duplicates=False, silent=False):
        """

        :param notnull_threshold: int or float. If int, it is the smallest number of not null values. If float, it is
                                  the smallest ratio of not null values.
        :param notsame_threshold: int or float. If int, it is the smallest number of not same values. If float, it is
                                  the smallest ratio of not same values.
        :param drop_duplicates:  default False. whether drop duplicates cases or not.
        :param silent: default False. whether silent print info.
        """
        self.notnull_threshold = notnull_threshold
        self.notsame_threshold = notsame_threshold
        self.drop_duplicates = drop_duplicates
        self.silent = silent

    def fit(self, df):
        df_desc = desc_df(df)
        if self.notnull_threshold > 1:
            self.notnan_ratio = self.notnull_threshold / df.shape[0]
        else:
            self.notnan_ratio = self.notnan_threshold
            self.notnull_threshold = int(df.shape[0] * self.notnan_ratio)
        self.nan_cols = df_desc.index[df_desc.notnull_ratio < self.notnan_ratio]

        if self.notsame_threshold > 1:
            self.same_ratio = 1 - self.notsame_threshold / df.shape[0]
        else:
            self.same_ratio = 1 - self.notsame_threshold
            self.notsame_threshold = int(df.shape[0] * self.notsame_threshold)
        self.same_cols = df_desc.index[df_desc.same_ratio > self.same_ratio]

        df = df.drop(self.same_cols, axis=1)
        df = df.drop(self.nan_cols, axis=1)

        n = df.shape[0]
        if self.drop_duplicates:
            df = df.drop_duplicates()
        m = df.shape[0]
        self.dup_num = n - m

        if not self.silent:
            print('columns which all values are nan: ', df_desc.index[df_desc.diff_values_num == 0])
            print('columns which all values are the same: ', df_desc.index[df_desc.diff_values_num == 1])
            print('columns which notnull values\' num below %s and nan ratio beyond %s: ' % (
                self.nan_threshold, self.nan_ratio), self.nan_cols.values)
            print('columns which not same values\' num below %s and same ratio beyond %s: ' % (
                self.same_threshold, self.same_ratio), self.same_cols.values)
            print('%s cases is duplicates.' % (n - m))

        self.df_desc = df_desc

    def transform(self, df):
        df = df.drop(self.same_cols, axis=1)
        df = df.drop(self.nan_cols, axis=1)

        n = df.shape[0]
        if self.drop_duplicates:
            df = df.drop_duplicates()
        m = df.shape[0]

        if not self.silent:
            print('%s cases is duplicates.' % (n - m))

        return df


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

    return df, df_nan


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
