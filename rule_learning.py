import pandas as pd
import numpy as np
from sklearn import feature_selection
from collections import defaultdict
import os


def chi2_calc(df_rule, labels, columns=None, save=True, path='datasource/chi2_result/', encoding='gbk'):
    if columns is None:
        columns = [c for c in df_rule.columns if c not in labels]

    dic_df = {}
    for t in labels:
        dic = defaultdict(dict)
        l = np.array((df_rule[t] > 0).astype(int))
        for c in columns:
            dic['命中样本数'][c] = df_rule[c].sum()
            dic['命中样本占比'][c] = df_rule[c].mean()
            dic['命中样本%s逾期率' % t][c] = l[np.where(df_rule[c] == 1)].mean()
            dic['未命中样本%s逾期率' % t][c] = l[np.where(df_rule[c] == 0)].mean()
            dic['总样本%s逾期率' % t][c] = l.mean()
            dic['p值'][c] = feature_selection.chi2(df_rule[c].reshape(-1, 1), l)[1][0]

        df_p = pd.DataFrame(dic)
        df_p = df_p[['总样本%s逾期率' % t, '命中样本%s逾期率' % t, '未命中样本%s逾期率' % t, 'p值', '命中样本占比', '命中样本数']].sort_values('p值')

        dic_df[f'rule_chi2_{t}'] = df_p

        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            for k, df_p in dic_df.items():
                # df_p.to_pickle(path + f'{k}.pkl')
                df_p.to_csv(f'{k}.csv', encoding=encoding)

    return dic_df
