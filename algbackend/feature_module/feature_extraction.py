import numpy as np
import pandas as pd
from pandas.api.types import is_timedelta64_ns_dtype
import json
import pandas as pd


def process_history(df_history, rid='apply_risk_id', created_at='created_at', pre_rid='pre_rid',
                    pre_created_at='pre_created_at', pre_finish_at='pre_finish_at',
                    pre_expect_at='pre_expect_at', pre_now_period='pre_now_period', pre_total_period='pre_total_period',
                    order='order', yuqi_day='yuqi_day', diff_date='diff_date'):
    df_history = df_history[
        (df_history[created_at] > df_history[pre_finish_at]) & (df_history[pre_finish_at].str[:10] != '1000-01-01')]
    df_history = df_history.sort_values(pre_now_period).drop_duplicates(pre_rid, keep='last')
    df_history = df_history[df_history[pre_now_period] == df_history[pre_total_period]]

    df_history[yuqi_day] = (pd.to_datetime(df_history[pre_finish_at].str[:10]) -
                            pd.to_datetime(df_history[pre_expect_at].str[:10])).map(lambda s: s.days)
    df_history[order] = df_history.groupby(rid)[pre_finish_at].rank(ascending=False)

    df_history = df_history.sort_values([rid, pre_created_at], ascending=False)
    df_history['post_pre_created_at'] = [np.nan] + list(df_history[pre_created_at][:-1])
    df_history['post_pre_created_at'][df_history.order == 1] = df_history[created_at][df_history.order == 1]
    df_history[diff_date] = (pd.to_datetime(df_history['post_pre_created_at'].str[:10]) -
                             pd.to_datetime(df_history[pre_finish_at].str[:10])).map(lambda s: s.days)
    return df_history


def latest(t):
    def __name__():
        return 'latest'

    return list(t)[0]


def spec_mean(t, spec=0):
    def __name__():
        return 'spec_mean'

    return np.mean(np.array(t) > spec)


def spec_sum(t, spec=0):
    def __name__():
        return 'spec_sum'

    return np.sum(np.array(t) > spec)


def stats_feature_extraction(t, span=None, stats=None, smooth=None):
    """
    适用于历史逾期天数，历史借贷间隔序列
    :param t: pd.Series. 按申请时间倒序
    :param span: 跨度， None|int。
    :param stats: 统计量， default None
    :param smooth: 平滑数
    :return:
    """
    name = t.name
    t = list(t)

    if span is not None:
        t = t[:span]
    else:
        span = 'all'

    if smooth is not None:
        t.append(smooth)

    if stats is None:
        stats = ['latest', 'np.min', 'np.max', 'np.mean', 'np.median', 'np.std', 'np.sum', 'len', 'spec_mean',
                 'spec_sum']

    if len(t) == 0:
        return json.dumps(
            {name + '_span_' + str(span) + '_' + eval(stats[i]).__name__: np.nan for i in range(len(stats))})

    return json.dumps({name + '_span_' + str(span) + '_' + eval(stats[i]).__name__: float(eval(stats[i])(t)) for i in
                       range(len(stats))})


def time_feature_extraction(t):
    """
    适用于历史订单发生时间，还款时间序列，得到历史订单发生时间的各小时/星期出现次数
    :param t: pd.Series type=datetime
    :return:
    """
    name = t.name

    t = pd.to_datetime(t)

    t1 = list(t.map(lambda s: s.hour))
    dic = {name + '_' + 'hour_' + str(i): t1.count(i) for i in range(24)}

    t2 = list(t.map(lambda s: s.weekday()))
    dic.update({name + '_' + 'weekday_' + str(i): t2.count(i) for i in range(7)})

    return json.dumps(dic)


class HistoryFeatureExtraction:
    def __init__(self, span_list=None, stats=None, smooth_f=True):
        if span_list is None:
            self.span_list = [3, None]
        else:
            self.span_list = span_list
        self.stats = stats
        self.smooth_f = smooth_f

    def fit(self, df, rid='apply_risk_id', created_at='created_at', pre_rid='pre_rid',
            pre_created_at='pre_created_at', pre_finish_at='pre_finish_at',
            pre_expect_at='pre_expect_at', pre_now_period='pre_now_period', pre_total_period='pre_total_period',
            order='order', yuqi_day='yuqi_day', diff_date='diff_date'):
        self.kwargs = {'rid': 'apply_risk_id', 'created_at': 'created_at', 'pre_rid': 'pre_rid',
                       'pre_created_at': 'pre_created_at', 'pre_finish_at': 'pre_finish_at',
                       'pre_expect_at': 'pre_expect_at', 'pre_now_period': 'pre_now_period',
                       'pre_total_period': 'pre_total_period',
                       'order': 'order', 'yuqi_day': 'yuqi_day', 'diff_date': 'diff_date'}
        self.rid = rid
        self.created_at = created_at
        self.pre_rid = pre_rid
        self.pre_created_at = pre_created_at
        self.pre_finish_at = pre_finish_at
        self.pre_expect_at = pre_expect_at
        self.pre_now_period = pre_now_period
        self.pre_total_period = pre_total_period
        self.order = order
        self.yuqi_day = yuqi_day
        self.diff_date = diff_date
        df = process_history(df, rid=rid, created_at=created_at, pre_rid=pre_rid,
                             pre_created_at=pre_created_at, pre_finish_at=pre_finish_at,
                             pre_expect_at=pre_expect_at, pre_now_period=pre_now_period,
                             pre_total_period=pre_total_period, order=order, yuqi_day=yuqi_day, diff_date=diff_date)
        if self.smooth_f is None:
            self.yq_smooth = None
            self.span_smooth = None
        else:
            self.yq_smooth = df[yuqi_day].median()
            self.span_smooth = df[diff_date].median()

    def transform(self, df):
        df = process_history(df, **self.kwargs)
        df = df.sort_values([self.rid, self.order])

        df_feature = pd.DataFrame()

        for c in [self.diff_date, self.yuqi_day]:
            if c == self.diff_date:
                smooth = self.span_smooth
            else:
                smooth = self.yq_smooth
            for span in self.span_list:
                df_feature = pd.concat([df_feature, df.groupby(self.rid)[c].agg(
                    lambda s: stats_feature_extraction(s, span=span, smooth=smooth)).apply(
                    lambda s: pd.Series(json.loads(s)))], axis=1)
        for c in [self.pre_created_at, self.pre_finish_at]:
            df_feature = pd.concat(
                [df_feature,
                 df.groupby(self.rid)[c].agg(time_feature_extraction).apply(lambda s: pd.Series(json.loads(s)))],
                axis=1)
        return df_feature


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


class TimeSpanFeatureExtraction(object):
    def __init__(self, smooth=None, sort=True, margins=True, time_unit='days', col_name_starts='',
                 extract_func=span_feature_extraction, stats=None, fill_value=True):
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
        self.fill_value = fill_value

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
            if self.fill_value:
                df_feature.fillna(df_feature.mean(), inplace=True)
        else:
            df_feature = pd.DataFrame()

        # 类内调用类方法，递归
        if self.hue is not None:
            for v in df[self.hue].unique():
                tmp = df[df[self.hue] == v]
                fe = TimeSpanFeatureExtraction(col_name_starts=self.col_name_starts + '_' + str(v))
                fe.fit(tmp, self.col, self.by, hue=None)
                df_feature = pd.concat([df_feature, fe.transform(tmp)], axis=1)

        return df_feature


if __name__ == '__main__':
    df = pd.read_pickle('../test_data/time_feature.pkl')
    fe = TimeSpanFeatureExtraction()
    fe.fit(df, '查询时间', 'apply_id', '主动查询用户信息的机构类型')
    t2 = fe.transform(df)
