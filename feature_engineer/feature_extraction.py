import numpy as np
import pandas as pd


def time_span_feature_extraction(t, stats=None, smooth=0, time_unit='days'):
    """
    得到历史订单发生时间间隔的统计量
    :param t: datetime
    :param stats:
    :param smooth:
    :param time_unit:
    :return:
    """
    if stats is None:
        stats = ['latest', 'min', 'max', 'mean', 'median', 'var', 'sum']

    t = np.diff(sorted(t))

    if len(t) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if time_unit == 'days':
        t = [i.days for i in t]
    elif time_unit == 'seconds':
        t = [i.seconds for i in t]
    else:
        raise Exception('param time_unit must be days or seconds!')

    t.insert(0, smooth)
    return t[-1], min(t), max(t), np.mean(t), np.median(t), np.var(t), np.sum(t)


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
