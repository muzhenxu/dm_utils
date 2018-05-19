import pandas as pd
import numpy as np
import random


class Psi(object):
    def __init__(self, max_th=10, min_th=-10, smooth=0.0001):
        self.max_th = max_th
        self.min_th = min_th
        self.smooth = smooth

    def get_cutpoints(self, expect_score, q=10):
        expect_score = pd.Series(expect_score)

        self.bins = self.qcut(expect_score, q=q)
        self.expect_dist = pd.DataFrame(
            pd.cut(expect_score, bins=self.bins, include_lowest=True).value_counts()).reset_index()
        self.expect_dist.columns = ['cat', 'expect_score']

    def get_psi(self, actual_score):
        actual_score = pd.Series(actual_score)

        s = pd.cut(actual_score, bins=self.bins, include_lowest=True).value_counts() + self.smooth
        s.columns = ['actual_score']

        self.dist = self.expect_dist.copy()
        self.dist['actual_score'] = self.dist.cat.map(s)

        self.dist['expect_score_ratio'] = self.dist.expect_score / self.dist.expect_score.sum()
        self.dist['actual_score_ratio'] = self.dist.actual_score / self.dist.actual_score.sum()

        lg = np.log(self.dist.actual_score_ratio / self.dist.expect_score_ratio)
        lg[lg > self.max_th] = self.max_th
        lg[lg < self.min_th] = self.min_th
        p = np.sum((self.dist.actual_score_ratio - self.dist.expect_score_ratio) * lg)

        return p

    @staticmethod
    def qcut(l, q=10, precision=8):
        _, cut_points = pd.qcut(l, q=q, retbins=True, precision=precision, duplicates='drop')

        cut_points = list(cut_points)

        cut_points.insert(0, 0)
        cut_points[-1] = np.inf
        return cut_points


if __name__ == '__main__':
    expect_data = [random.uniform(0, 1) for i in range(1000)]
    actual_data = [random.uniform(0, 1) for i in range(3000)]
    psi = Psi()
    psi.get_cutpoints(expect_data)
    print(psi.get_psi(actual_data))
