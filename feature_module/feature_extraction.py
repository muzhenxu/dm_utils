import numpy as np
import pandas as pd
from pandas.api.types import is_timedelta64_ns_dtype


def span_feature_extraction(t, stats=None):
    """
    得到历史订单发生时间间隔的统计量
    :param t: 时间间隔
    :param stats:
    :param smooth:
    :param time_unit:
    :return:
    """
    # TODO: stats应该通过eval发挥函数作用，不然是硬编码，不好维护
    if stats is None:
        stats = ['latest', 'min', 'max', 'mean', 'median', 'std', 'sum', 'len']

    if len(t) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return t.iloc[-1], np.min(t), np.max(t), np.mean(t), np.median(t), np.std(t), np.sum(t), len(t)


def time_hour_feature_extraction(t):
    """
    得到历史订单发生时间的各小时出现次数
    :param t:
    :return:
    """
    t = [i.hour for i in t]
    l = []
    for i in range(24):
        l.append(t.count(i))
    return tuple(l)


def delay_stats_feature_extraction(t, smooth=0):
    """
    得到历史订单逾期情况统计量
    :param t:
    :param smooth:
    :return:
    """
    t = list(t)

    if len(t) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    t.insert(0, smooth)
    return t[-1], min(t), max(t), np.mean(t), np.median(t), np.var(t), np.mean(t > 0), np.sum(t > 0)


class TimeSpanFeatureExtraction(object):
    def __init__(self, smooth=None, sort=True, margins=True, time_unit='days', col_name_starts='', extract_func=span_feature_extraction, stats=None):
        """

        :param col:
        :param by:
        :param hue:
        :param smooth: False|None|numeric. if FALSE，don't smooth. if None, use mean to smooth.
        :param sort:
        :param margins:
        :param time_unit:
        """

        self.smooth = smooth
        self.sort = sort
        self.margins = margins
        self.time_unit = time_unit
        self.col_name_starts = col_name_starts
        self.extract_func = extract_func

        if stats is None:
            self.stats = ['latest', 'min', 'max', 'mean', 'median', 'std', 'sum', 'len']
        else:
            self.stats = stats

    def fit(self, df, col, by, hue=None):
        self.col = col
        self.by = by
        self.hue = hue

        if (self.hue is None) & (self.margins is False):
            raise Exception('hue and margins can\'t false all!')

        df.index = range(df.shape[0])

        if self.sort:
            df = df.sort_values(self.col)

        df[self.col] = df[self.col].apply(pd.to_datetime, errors='ignore')
        df['diff_cols'] = df.groupby(self.by)[self.col].diff()

        df = df.dropna(subset=['diff_cols'])

        if is_timedelta64_ns_dtype(df['diff_cols']):
            if self.time_unit == 'days':
                df['diff_cols'] = df['diff_cols'].map(lambda s: s.days)
            elif self.time_unit == 'seconds':
                df['diff_cols'] = df['diff_cols'].map(lambda s: s.seconds)

        if self.smooth is None:
            self.smooth = df['diff_cols'].mean()

    def transform(self, df):
        df.index = range(df.shape[0])

        if self.sort:
            df = df.sort_values(self.col)

        df[self.col] = df[self.col].apply(pd.to_datetime, errors='ignore')
        df['diff_cols'] = df.groupby(self.by)[self.col].diff()

        if self.smooth is not False:
            df['diff_cols'].fillna(self.smooth)

        if is_timedelta64_ns_dtype(df['diff_cols']):
            if self.time_unit == 'days':
                df['diff_cols'] = df['diff_cols'].map(lambda s: s.days)

        if self.margins:
            # TODO: 利用stats作为span_feature_extraction的入参
            df_feature = df.groupby(self.by)['diff_cols'].agg(span_feature_extraction).apply(pd.Series)
            df_feature.columns = [self.col_name_starts + '_' + i for i in self.stats]
            df_feature.fillna(df_feature.mean(), inplace=True)
        else:
            df_feature = pd.DataFrame()

        # 类内调用类方法，递归
        if self.hue is not None:
            for v in df[self.hue].unique():
                tmp = df[df[self.hue] == v]
                fe = TimeSpanFeatureExtraction(col_name_starts=self.col_name_starts + str(v))
                fe.fit(tmp, self.col, self.by, hue=None)
                df_feature = pd.concat([df_feature, fe.transform(tmp)], axis=1)

        return df_feature

if __name__ == '__main__':
    df = pd.read_pickle('../test_data/time_feature.pkl')
    fe = TimeSpanFeatureExtraction()
    fe.fit(df, '查询时间', 'apply_id', '主动查询用户信息的机构类型')
    t2 = fe.transform(df)