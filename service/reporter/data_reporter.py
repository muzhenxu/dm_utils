import pandas as pd
import numpy as np
from ..base_utils import *
from ...algbackend.feature_module import *

def get_max_same_count(c):
    try:
        return c.value_counts().iloc[0]
    except:
        return len(c)


def data_stats_report(df):
    """
    数据描述性统计
    :param df:
    :return:
    """
    t1 = df.describe().T
    t2 = get_missing_value_ratio(df)
    t3 = get_same_value_ratio(df)

    t = pd.concat([t1, t2, t3], axis=1)

    return t

def feature_evaluation_report(df, y, **kwargs):
    """
    单特征iv，auc，ks计算
    :param df:
    :param y:
    :param kwargs:
    :return:
    """
    df = pd.concat([df, y], axis=1)
    df.fillna(-999, inplace=True)
    res = feature_evaluation.feature_evaluation(df, [y.names], **kwargs)
    return res

