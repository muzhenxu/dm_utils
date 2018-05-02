import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from .continous_encoding import ContinousEncoder
from .category_encoding import CategoryEncoder
from ..base_utils import *

class FeatureTransformer():
    def __init__(self, cate_thr=20, cate_cut=0.95, nan_cut=0.95, same_cut=0.95, cate_method='CountEncoder',
                 cont_method='NothingEncoder'):
        self.cate_cut = cate_cut
        self.nan_cut = nan_cut
        self.same_cut = same_cut
        self.cate_thr = cate_thr
        self.cate_method = cate_method
        self.cont_method = cont_method

    def fit(self, df, label=None):
        df = df.apply(pd.to_numeric, errors='ignore')

        exclude_cols = find_nan_same_cate_cols(df, self.nan_cut, self.same_cut, self.cate_cut)
        df = df.drop(exclude_cols, axis=1)

        cols = df.nunique()[df.nunique() < self.cate_thr].index.values
        df[cols] = df[cols].astype(str)

        tmp = df.dtypes.map(is_numeric_dtype)
        self.continous_features = tmp[tmp].index.values
        self.categorial_features = tmp[~tmp].index.values

        if len(self.continous_features) > 0:
            # convert bool to int
            df_cont = (df[self.continous_features] * 1).apply(pd.to_numeric)

            self.cont_enc = ContinousEncoder(self.cont_method)
            self.cont_enc.fit(df_cont, label)

        if len(self.categorial_features) > 0:
            df_cate = df[self.categorial_features]
            self.cate_enc = CategoryEncoder(self.cate_method)
            self.cate_enc.fit(df_cate, label)

    def transform(self, df):
        if len(self.continous_features) > 0:
            df_cont = df[self.continous_features]
            df_cont = (df_cont.apply(pd.to_numeric, errors='ignore') * 1).apply(pd.to_numeric)
            df_cont = self.cont_enc.transform(df_cont)
        else:
            df_cont = pd.DataFrame()

        if len(self.categorial_features) > 0:
            df_cate = df[self.categorial_features].astype(str)
            df_cate = self.cate_enc.transform(df_cate)
        else:
            df_cate = pd.DataFrame()

        df = pd.concat([df_cont, df_cate], axis=1)
        return df


def find_nan_same_cate_cols(df, nan_cut=0.8, same_cut=0.8, cate_cut=0.9):
    """

    :param df:
    :param nan_cut:
    :param same_cut:
    :param cate_cut: 如果是二值变量，很容易就0.6，0.7
    :return:
    """
    df = df.apply(pd.to_numeric, errors='ignore')

    tmp = df.dtypes.map(is_numeric_dtype)
    categorial_features = tmp[~tmp].index.values

    # 寻找缺失值严重列
    tmp = get_missing_value_ratio(df)
    nan_cols = tmp[tmp > nan_cut].index.values

    # 寻找同值严重列
    tmp = get_same_value_ratio(df)
    same_cols = tmp[tmp > same_cut].index.values

    # 寻找不同值严重cate列
    tmp = df[categorial_features]
    tmp = tmp.nunique() / df.shape[0]
    cate_cols = tmp[tmp > cate_cut].index.values

    exclude_cols = set(list(nan_cols) + list(same_cols) + list(cate_cols))

    return exclude_cols
