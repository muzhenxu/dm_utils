import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import bisect
from collections import Counter


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

    def __init__(self):
        super(labelencoder, self).__init__()

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

    def __init__(self, unseen_values=1):
        self.unseen_values = 1

    def fit(self, X, y=None):
        """
        :param X: array-like of shape (n_samples,)
        :param y: None
        :return:
        """
        Counter(X)

    def transform(self, X):
        """
        :param X: array-like of shape (n_samples,)
        :return:
        """


# TODO: hashencoder

from sklearn.feature_extraction.text import CountVectorizer


class CategoryEncoder(object):
    def __init__(self, method='onehot'):
        self.method = method

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass
