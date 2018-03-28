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
import math


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


def get_cut_points(X, y=None, bins=10, binning_method='dt', precision=8, **kwargs):
    if binning_method == 'cut':
        _, cut_points = pd.cut(X, bins=bins, retbins=True, precision=precision)
    elif binning_method == 'qcut':
        _, cut_points = pd.qcut(X, q=bins, retbins=True, duplicates='drop', precision=precision)
    elif binning_method == 'dt':
        cut_points = dt_cut_points(X, y, **kwargs)
    else:
        raise ValueError("binning_method: '%s' is not defined." % binning_method)

    if binning_method != 'dt':
        cut_points = cut_points[1:-1]

    cut_points = list(cut_points)
    cut_points.append(np.inf)
    cut_points.insert(0, -np.inf)

    return cut_points


def woe(x, y, woe_min=-20, woe_max=20):
    # TODO: woe_min&woe_max设置是否合理？
    """

    :param x: array
    :param y: array
    :return:
    """
    x = np.array(x)
    y = np.array(y)

    pos = (y == 1).sum()
    neg = (y == 0).sum()

    dmap = {}

    for k in np.unique(x):
        indice = np.where(x == k)
        pos_r = (y[indice] == 1).sum() / pos
        neg_r = (y[indice] == 0).sum() / neg

        if pos_r == 0:
            woe1 = woe_min
        elif neg_r == 0:
            woe1 = woe_max
        else:
            woe1 = math.log(pos_r / neg_r)

        dmap[k] = woe1

    return dmap


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


class binningencoder(object):
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
        :param labels: category names for bins
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

        self.cut_points = get_cut_points(X, y, self.bins, self.binning_method, **self.kwargs)

        if self.interval:
            self.labels = np.arange(len(self.cut_points) - 1)

    def transform(self, X):
        X = pd.cut(X, bins=self.cut_points, labels=self.labels).astype(int)
        return X


class woeencoder(object):
    def __init__(self, bins=10, binning_method='dt', woe_min=-20, woe_max=20, **kwargs):
        self.bins = bins
        self.binning_method = binning_method
        self.kwargs = kwargs
        self.woe_min = woe_min
        self.woe_max = woe_max

    def fit(self, X, y):
        self.cut_points = get_cut_points(X, y, self.bins, self.binning_method, **self.kwargs)

        self.dmap = woe(X, y, self.woe_min, self.woe_max)
        self.base_p = y.mean()

    def transform(self, X):
        X = pd.cut(X, bins=self.cut_points)
        X = np.array([self.dmap[i] if i in self.dmap.keys() else self.base_p for i in X])
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

    def fit(self, X, y):
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


def CategoryEncoder(method='countencoder', **kwargs):
    """
    contain these method: onehotencoder, labelencoder, targetencoder, countencoder(default)

    :param method: onehotencoder, labelencoder, targetencoder, countencoder(default)
    :return:
    """
    return eval(method)(**kwargs)


if __name__ == '__main__':
    enc = CategoryEncoder()
    enc.fit(np.array(['a', 'c', 'd', 'a', 'a', 'd']))
    enc.transform(np.array(['f', 'c', 'd']))
