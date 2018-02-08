#!/usr/bin/python
# -*- coding: utf-8 -*-
import math

def hold_out(df,train_size=0.8,suffle=True,random_state = 2018):
    """按照传入参数分割原DataFrame 为两个无交集的 DataFrame。

    这个方法会对原始df 做无交集分割处理，根据传入的比例，是否洗牌，和初始种子
    分割原数据为2个数据集后输出

    Parameters（参数）
    ----------
    df ： DataFrame 传入待分解数据集

    train_size ： 分出的train 比例
        如果最后是分数则会向下取整（200.9 取 200）

    suffle ： 是否需要对原始数据集进行洗牌
        默认为True 如果传入False则不洗牌，按传入顺序iloc 分割

    random_state ： 初始随机种子
        用于复现已经跑出的结果，避免2此执行获得不同的结果值

    return:
        2个df ：train 和 test
        ！输出的df 中行顺序标识可能与原df 行标识不一致
    """
    if suffle:
        dsf=df.sample(frac=1,random_state=random_state).reset_index(drop=True)
    else:
        dsf = df.reset_index(drop=True)
    l = len(df)
    ts=(int)(l*train_size)
    train = dsf.iloc[:ts]
    test = dsf.iloc[ts:]
    return train,test
