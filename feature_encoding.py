"""
Use this class to process categorical variables.
document: https://www.slideshare.net/HJvanVeen/feature-engineering-72376750
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import bisect
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class labelencoder(LabelEncoder):
    """
    sklearn.preprocess.LabelEncoder can't process values which don't appear in fit label encoder.
    this method can process this problem. Replace all unknown values to a certain value, and encode this
    value to 0.

    Attributes
    ----------
    like sklearn.preprocess.LabelEncoder

    Example
    -------
    enc = labelencoder()
    enc.fit(['a','b','c'])
    enc.transform(['a','v','d'])
    Out: array([1, 0, 0])

    """

    # if don't explicitly specify __init__, class will share it's parent class's __init__ params.
    # def __init__(self):
    #     super(labelencoder, self).__init__()

    def fit(self, X, y=None):
        """
        :param X: array-like of shape (n_samples,)
        :param y: None
        :return:
        """
        l = list(np.unique(X))
        t1 = '<unknown>'
        t2 = -999
        while (t1 in l):
            t1 = t1 + '*'
        while (t2 in l):
            t2 -= t2

        le = LabelEncoder(**self.get_params())
        le.fit(X)

        le_classes = le.classes_.tolist()
        try:
            bisect.insort_left(le_classes, t1)
            self.unknown = t1
        except:
            bisect.insort_left(le_classes, t2)
            self.unknown = t2
        le.classes_ = le_classes
        self.encoder = le

    def transform(self, X):
        """
        :param X: array-like of shape (n_samples,)
        :return:
        """
        X = [s if s in self.encoder.classes_ else self.unknown for s in X]
        return self.encoder.transform(X)


class onehotencoder(OneHotEncoder):
    """
    sklearn.preprocess.OnehotEncoder only can process numerical values.
    this method can process str.

    Attributes
    ----------
    like sklearn.preprocess.OneHotEncoder

    Example
    -------
    enc = onehotencoder(sparse=False)
    enc.fit(['a','b','c'])
    enc.transform(['a','v','d'])
    Out:
    array([[ 1.,  0.,  0.],
       [ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])


    """

    # def __init__(self):
    #     super(onehotencoder, self).__init__()

    def fit(self, X, y=None):
        """
        :param X: array-like of shape (n_samples,)
        :param y: None
        :return:
        """
        le = labelencoder()
        le.fit(X)
        self.le = le

        X = self.le.transform(X)

        # below codes can share the init params, but onehot will be not a instance.so will haven't its attributes.
        # onehot = OneHotEncoder
        # onehot.fit(self, X.reshape(-1, 1))
        # self.encoder.transform(self, X.reshape(-1, 1))

        onehot = OneHotEncoder(**self.get_params())
        onehot.fit(X.reshape(-1, 1))

        self.encoder = onehot

    def transform(self, X):
        """
        :param X: array-like of shape (n_samples,)
        :return:
        """
        X = self.le.transform(X)
        return self.encoder.transform(X.reshape(-1, 1))


class countencoder(object):
    """
    count encoding: Replace categorical variables with count in the train set.
    replace unseen variables with 1.
    Can use log-transform to be avoid to sensitive to outliers.
    Only provide log-transform with base e, because I think it's enough.


    Attributes
    ----------
    dmap: a collections.Counter(which like dict) map variable's values to its frequency.

    Example
    -------
    enc = countencoder()
    enc.fit(['a','b','c', 'b', 'c', 'c'])
    enc.transform(['a','c','b'])
    Out:
    array([ 0.        ,  1.09861229,  0.69314718])

    """

    def __init__(self, unseen_values=1, log_transform=True, smoothing=1):
        self.unseen_values = unseen_values
        self.log_transform = log_transform
        self.smoothing = 1

    def fit(self, X, y=None):
        """
        :param X: array-like of shape (n_samples,)
        :param y: None
        :return:
        """
        self.dmap = Counter(X)

    def transform(self, X):
        """
        :param X: array-like of shape (n_samples,)
        :return:
        """
        # TODO: maybe use pd.Series with replace can faster. should test.
        X = np.array([self.dmap[i] + self.smoothing if i in self.dmap.keys() else self.unseen_values for i in X])
        if self.log_transform:
            X = np.log(X)
        return X


class targetencoder(object):
    """
    this method uses to encode variables by target.
    Only support binary classification and regression.
    Form of stacking: single-variable model which outputs average target.

    use m-estimate to smooth.
    use normal to random value.

    Attributes
    ----------
    dmap: a dict map variables to its average target with smooth and random.
    base_p: target mean

    Example
    -------
    enc = targetencoder()
    enc.fit(np.array(['a','b','c', 'b', 'c', 'c']), np.array([1, 0, 1, 1, 0, 1]))
    enc.transform(np.array(['a','c','b']))
    Out:
    array([ 1.03627629,  0.58939665,  0.55091546])

    """

    def __init__(self, random_noise=0.05, smoothing=0.1, random_seed=10):
        self.random_noise = random_noise
        self.smoothing = smoothing
        self.random_seed = random_seed

    def fit(self, X, y=None):
        # TODO: add if condition to judge X is continous or binary.
        # TODO: Is it necessary to make sure values which add random keep theres order? And does control values less than 1 and more than 0?
        if y is None:
            raise Exception('encoder need valid y label.')

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(X)
        np.random.seed(self.random_seed)
        self.bias = np.random.normal(0, self.random_noise, len(self.classes_))
        self.dmap = {}
        self.base_p = y.mean()
        for i, key in enumerate(self.classes_):
            l = y[X == key]
            p = (sum(l) + self.smoothing * len(l) * self.base_p) / (len(l) + self.smoothing * len(l))
            p += self.bias[i]
            self.dmap[key] = p

    def transform(self, X):
        X = np.array([self.dmap[i] if i in self.dmap.keys() else self.base_p for i in X])
        return X


# class CategoryEncoder(object):
#     def __init__(self, method='onehotencoder'):
#         self.method = method
#
#     def fit(self, X, y=None):
#         self.enc = eval(self.method)()
#         self.enc.fit(X, y)
#
#     def transform(self, X):
#         return self.enc.transform(X)

def CategoryEncoder(method='countencoder'):
    """
    contain these method: onehotencoder, labelencoder, targetencoder, countencoder(default)

    :param method: onehotencoder, labelencoder, targetencoder, countencoder(default)
    :return:
    """
    return eval(method)()


def dt_cut_points(x, y, max_depth=4, min_samples_leaf=0.05, max_leaf_nodes=None, random_state=7):
    """
    A decision tree method to bin continuous variable to categorical one.
    :param x: The training input samples
    :param y: The target values
    :param max_depth: The maximum depth of the tree
    :param min_samples_leaf: int, float, The minimum number of samples required to be at a leaf node
    :param max_leaf_nodes: Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    :return: The list of cut points
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                random_state=random_state)
    dt.fit(np.array(x).reshape(-1, 1), np.array(y))
    th = dt.tree_.threshold
    f = dt.tree_.feature

    # 对于没有参与分裂的节点，dt默认会给-2,所以这里要根据dt.tree_.feature把-2踢掉
    return sorted(th[np.where(f != -2)])


class BinningEncoder(object):
    """
    convert continuous varaible to discrete variable.

    Example
    -------
    enc = BinningEncoder()
    enc.fit(np.random.rand(100), np.random.randint(0, 2, 100))
    enc.transform(np.random.rand(100))
    """
    def __init__(self, bins=10, binning_method='dt', labels=None, interval=True, **kwargs):
        """

        :param bins:
        :param binning_method: have three methods: 'dt' which uses decision tree, 'cut' which cuts data by the equal intervals,
        'qcut' which cuts data by the equal quantity. default is 'dt'. if y is None, default auto changes to 'qcut'.
        :param labels:
        :param kwargs: params for decision tree.
        :param interval: if interval is True, param labels is invalid.
        """
        self.bins = bins
        self.labels = labels
        self.interval = interval
        self.binning_method = binning_method
        self.kwargs = kwargs

    def fit(self, X, y=None):
        if y is None:
            self.binning_method = 'qcut'

        if self.labels is not None and len(self.labels) != self.bins:
            raise ValueError('the length of labels must be equal to bins.')

        if self.binning_method == 'cut':
            _, self.cut_points = pd.cut(X, bins=self.bins, retbins=True)
        elif self.binning_method == 'qcut':
            _, self.cut_points = pd.qcut(X, q=self.bins, retbins=True, duplicates='drop')
        elif self.binning_method == 'dt':
            self.cut_points = dt_cut_points(X, y, **self.kwargs)
        else:
            raise ValueError("binning_method: '%s' is not defined." % self.binning_method)

        if self.binning_method != 'dt':
            self.cut_points = self.cut_points[1:-1]

        self.cut_points = list(self.cut_points)
        self.cut_points.append(np.inf)
        self.cut_points.insert(0, -np.inf)

        if self.interval:
            self.labels = np.arange(len(self.cut_points)-1)

    def transform(self, X):
        X = pd.cut(X, bins=self.cut_points, labels=self.labels)
        return X


# TODO: move to feature_encoding.py
def woe_translate(df_origin, labels, columns=None, onehot=True, del_origin_columns=False):
    """
    woe encoder
    :param df_origin:
    :param columns:
    :param labels: type: list
    :return:
    """
    df = df_origin.copy()
    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    woe = WOE()
    mdlp = MDLP()

    dic_cut_points = {}
    dic_woe = {}
    dic_iv = defaultdict(dict)
    dic_nmi = defaultdict(dict)

    for c in columns:
        if df[c].nunique() <= 1:
            continue
        for t in labels:
            dic_cut_points[c] = mdlp.cut_points(np.array(df[c]), np.array(df[t]))
            df['%s_%s_mdlp' % (c, t)] = mdlp.discretize_feature(df[c], dic_cut_points[c])
            dic_woe[c], dic_iv[t][c] = woe.woe_single_x(df['%s_%s_mdlp' % (c, t)], df[t])
            dic_nmi[t][c] = mr.normalized_mutual_info_score(df['%s_%s_mdlp' % (c, t)], df[t])
            df['%s_%s_woe' % (c, t)] = df['%s_%s_mdlp' % (c, t)].replace(dic_woe[c].keys(), dic_woe[c].values())
            if onehot:
                df = pd.concat([df, pd.get_dummies(df['%s_%s_mdlp' % (c, t)], prefix='%s_%s_mdlp' % (c, t))], axis=1)

            if del_origin_columns:
                del df['%s_%s_mdlp' % (c, t)]
            print(c, t)

    if del_origin_columns:
        df = df.drop(columns, axis=1)

    df_iv = pd.DataFrame(dic_iv)
    df_nmi = pd.DataFrame(dic_nmi)
    df_iv.columns = [['IV'] * df_iv.shape[1], df_iv.columns]
    df_nmi.columns = [['NMI'] * df_nmi.shape[1], df_nmi.columns]
    df_iv_nmi = pd.concat([df_iv, df_nmi], axis=1)

    return df, df_iv_nmi, dic_cut_points, dic_woe

# TODO: hashencoder
# import hashlib
# class HashingEncoder(object):
#     def __init__(self):
#         self.cols_set = []
#         self.unknown_type = None
#
#     def fit(self, X, col=None, n_components=5, hashing_method='md5'):
#         """
#         :param X: array-like of shape (n_samples,)
#         :param y: None
#         :return:
#         """
#         if n_components <= 0:
#             raise ValueError('n_components shout be greater than 0.')
#
#         if not col:
#             col = 'hh'
#         self.col = col
#         self.n_components = n_components
#         self.hashing_method = hashing_method
#
#         self.cols_set = list(np.unique(X))
#         self.unknown_type = '<unknown>'
#         while (self.unknown_type in self.cols_set):
#             self.unknown_type += '*'
#         return self
#
#     def transform(self, X):
#         """
#         :param X: array-like of shape (n_samples,)
#         :return:
#         """
#         X_tmp = [_  if _ in self.cols_set else self.unknown_type  for _ in X]
#         X_tmp = pd.DataFrame(X_tmp, columns=[self.col])
#         return self.__hash_col(X_tmp)
#
#
#     def __hash_col(self, df):
#         """
#         :param df: dataframe of X
#         return:
#         """
#         cols = [f'{self.col}_{i}' for i in range(self.n_components)]
#         def xform(x):
#             tmp = np.zeros(self.n_components)
#             tmp[self.__hash(x) % self.n_components] = 1
#             return pd.Series(tmp, index=cols).astype(int)
#         df[cols] = df[self.col].apply(xform)
#         return df.drop(self.col, axis=1)
#
#     def __hash(self, string):
#         if self.hashing_method == 'md5':
#             return int(hashlib.md5(str(string).encode('utf-8')).hexdigest(), 16)
#         else:
#             raise ValueError('Hashing Method: %s Not Available. Please check that.' % self.hashing_method)


if __name__ == '__main__':
    enc = CategoryEncoder()
    enc.fit(np.array(['a', 'c', 'd', 'a', 'a', 'd']))
    enc.transform(np.array(['f', 'c', 'd']))
