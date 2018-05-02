# references:
# 1. df2libffm： https://github.com/chengstone/kaggle_criteo_ctr_challenge-/blob/master/ctr.ipynb
# 2. 并行化支持：https://github.com/totoruo/FfmEncoder/blob/master/FfmEncoder.py

import hashlib, math, os, subprocess
from multiprocessing import Process
from pandas.api.types import is_numeric_dtype
from collections import OrderedDict
import pandas as pd
from collections import defaultdict
import numpy as np


def hashstr(str, nr_bins=1e+6):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (int(nr_bins) - 1) + 1


class FfmEncoder1(object):
    """
    df2libffm, label必须放在df的第一列
    """

    def __init__(self, df, nthread=1):
        """
        一般情况下一个原特征对应一个field。
        :param field_names: list. 长度应该和原始特征个数一致
        :param nthread:
        """
        self.field_names = OrderedDict(df.apply(is_numeric_dtype)).items()[1:]
        self.nthread = nthread

    def gen_feats(self, row):
        feats = []
        for i, field in enumerate(self.field_names, start=1):
            value = row[i]  # row[0] is label
            if field[1]:
                key = value
            else:
                key = field[0] + '-' + str(value)
            feats.append(key)
        return feats

    def gen_hashed_fm_feats(self, feats):
        feats = ['{0}:{1}:1'.format(field, hashstr(feat, 1e+6)) for (field, feat) in feats]
        return feats

    def convert(self, df, path, i):
        lines_per_thread = math.ceil(float(df.shape[0]) / self.nthread)
        sub_df = df.iloc[i * lines_per_thread: (i + 1) * lines_per_thread]
        tmp_path = path + '_tmp_{0}'.format(i)
        with open(tmp_path, 'w') as f:
            for row in sub_df.values:
                feats = []
                for i, feat in enumerate(self.gen_feats(row)):
                    feats.append((i, feat))
                feats = self.gen_hashed_fm_feats(feats)
                f.write(str(int(row[0])) + ' ' + ' '.join(feats) + '\n')

    def parallel_convert(self, df, path):
        processes = []
        for i in range(self.nthread):
            p = Process(target=self.convert, args=(df, path, i))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def delete(self, path):
        for i in range(self.nthread):
            os.remove(path + '_tmp_{0}'.format(i))

    def cat(self, path):
        if os.path.exists(path):
            os.remove(path)
        for i in range(self.nthread):
            cmd = 'cat {svm}_tmp_{idx} >> {svm}'.format(svm=path, idx=i)
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()

    def transform(self, df, path):
        print('converting data......')
        self.parallel_convert(df, path)
        print('catting temp data......')
        self.cat(path)
        print('deleting temp data......')
        self.delete(path)
        print('transform done!')


class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """

    def __init__(self, continous_features, field_names=None):
        self.continous_features = continous_features
        if field_names is None:
            self.field_names = np.array(range(0, len(continous_features)))
        else:
            self.field_names = field_names

    def build(self, df, low_q=5, high_q=95, feats_min=None, feats_max=None):
        sub_df = df[self.continous_features]

        if feats_min is None:
            t_low = sub_df.apply(np.percentile, q=low_q)
            sub_df = sub_df[sub_df > t_low].fillna(t_low)
            self.min = sub_df.min()
        else:
            self.min = feats_min

        if feats_max is None:
            t_high = sub_df.apply(np.percentile, q=high_q)
            sub_df = sub_df[sub_df < t_high].fillna(t_high)
            self.max = sub_df.max()
        else:
            self.max = feats_max

    def gen(self, df):
        sub_df = df[self.continous_features]
        sub_df = (sub_df - self.min) / (self.max - self.min)
        sub_df = sub_df[sub_df > 0].fillna(0)
        sub_df = sub_df[sub_df < 1].fillna(1)

        tmp = pd.DataFrame([self.field_names] * sub_df.shape[0])
        tmp.columns = sub_df.columns
        tmp = tmp.astype(str)
        sub_df = sub_df.astype(str)
        sub_df = tmp + ':' + tmp + ':' + sub_df
        return sub_df


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, categorial_features, field_names=None, start_idx=0):
        self.categorial_features = categorial_features
        if field_names is None:
            self.field_names = np.array(range(start_idx, len(categorial_features) + start_idx))
        else:
            self.field_names = field_names
        self.start_idx = start_idx

    def build(self, df, cutoff=0):
        self.dicts = defaultdict(dict)

        unk = '<unk>'
        while df[self.categorial_features].isin([unk]).any().any():
            unk += unk
        self.unk = unk
        start_idx = self.start_idx

        for f, c in enumerate(self.categorial_features):
            tmp = df[c].value_counts()
            vocabs = [unk] + list(tmp[tmp >= cutoff].index.values)

            idx_list = np.array(range(len(vocabs))) + start_idx
            idx_list = [str(self.field_names[f]) + ':' + str(i) + ':1' for i in idx_list]

            self.dicts[c] = dict(zip(vocabs, idx_list))
            start_idx += len(vocabs)

    def gen(self, df):
        sub_df = df[self.categorial_features]

        for c in self.categorial_features:
            sub_df[c][~sub_df[c].isin(self.dicts[c].keys())] = self.unk
            sub_df[c] = sub_df[c].replace(self.dicts[c])

        return sub_df


# TODO: 并行化支持
class FFMEncoder(object):
    def __init__(self, nthread=1):
        self.nthread = nthread

    def fit(self, df, cutoff=0):
        tmp = df.dtypes.map(is_numeric_dtype)
        self.continous_features = tmp[tmp].index.values
        self.categorial_features = tmp[~tmp].index.values

        cfg = ContinuousFeatureGenerator(self.continous_features)
        cfg.build(df)
        self.cfg = cfg

        cdg = CategoryDictGenerator(self.categorial_features, start_idx=len(self.continous_features))
        cdg.build(df, cutoff)
        self.cdg = cdg

    def transform(self, df, label=None, path='datasource/data.ffm', save_file=True):
        """

        :param df:
        :param label: pd.Series or None. default None.
        :return:
        """
        continous_df = self.cfg.gen(df)
        category_df = self.cdg.gen(df)
        if label is None:
            libffm_df = pd.concat([continous_df, category_df], axis=1)
        else:
            libffm_df = pd.concat([label, continous_df, category_df], axis=1)
        if save_file:
            if os.path.dirname(path) != '':
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
            libffm_df.to_csv(path, sep=' ', header=False, index=False)
        return libffm_df


if __name__ == '__main__':
    cat1 = ['AAA', 'BBB', 'CCC']
    cat2 = ['DDD', 'EEE', 'FFF']
    label = [0, 1, 0]
    num1 = [1, 2, 3]
    num2 = [4, 5, 6]
    df = pd.DataFrame({'Label': label, 'cat1': cat1, 'cat2': cat2, 'num1': num1, 'num2': num2})

    fe = FFMEncoder()
    fe.fit(df.iloc[:, 1:])
    df2 = fe.transform(df.iloc[:, 1:], df['Label'])
