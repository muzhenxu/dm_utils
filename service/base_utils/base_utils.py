import pandas as pd
import numpy as np

def get_max_same_count(c):
    try:
        return c.value_counts().iloc[0]
    except:
        return len(c)

def get_same_value_ratio(df):
    t = df.apply(get_max_same_count) / df.shape[0]
    t.name = 'same_value'
    return t

def get_missing_value_ratio(df):
    t = df.isnull().mean()
    t.name = 'missing'
    return t