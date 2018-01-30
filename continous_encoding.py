# 装饰器 - 输入和返回数据统一格式
# TODO: transfer the output to the one-hot-encoder
class BinningEncoder(object):
    def __init__(self):
        self.cut_points = []
        self.labels = None

    def fit(self, X, n_components=10, binning_method='cut', labels=None):
        """
        :param X: array-like of shape (n_samples,)
        :param n_components: the number of bins
        :param binning_method: the method of the bin
        :pabel labels:
        :return:
        """
        # 等频 qcut
        # 等宽 cut
        # 决策树 dt
        self.n_components = n_components
        self.binning_method = binning_method
        self.labels = labels
        if self.labels and len(self.labels) != self.n_components:
            raise ValueError('Bin labels must be one fewer than the number of bin edges.')

        if self.binning_method == 'cut':
            try:
                r = pd.cut(X, bins=self.n_components, retbins=True)
                self.cut_labels = r[0]._get_categories().to_native_types().tolist()
                self.cut_points = r[1].tolist()
            except:
                raise Exception('Error in pd.cut')
        elif self.binning_method == 'qcut':
            try:
                r = pd.qcut(data, q=np.linspace(0, 1, self.n_components), precision=0, retbins=True, labels=labels)
                self.cut_labels = r[0]._get_categories().to_native_types().tolist()
                self.cut_points = r[1].tolist()
            except:
                raise Exception('Error in pd.qcut')
        elif self.binning_method == 'dt':
            pass
        else:
            raise ValueError("binning_method: '%s' is not defined." % self.binning_method)

        if np.inf not in self.cut_points:
            self.cut_labels.append(f'(%s, inf]' % self.cut_points[-1])
            self.cut_points.append(np.inf)
        if -np.inf not in self.cut_points:
            self.cut_labels.insert(0, f'(-inf, %s]' % self.cut_points[0])
            self.cut_points.insert(0, -np.inf)


    def transform(self, X, one_hot=False):
        """
        :param X: array-like of shape (n_samples,)
        :param one_hot: return one_hot type of X or not
        :return:
        """
        if self.cut_points == []:
            raise ValueError('cut_points is empty.')
        try:
            r = pd.cut(X, bins=self.cut_points, labels=self.labels, retbins=True)[0]
        except:
            raise Exception('Error in pd.cut')
        if not one_hot:
            # labels = list(pd.Series(r.get_values()))                                            # interval type for sorting
            return pd.DataFrame([X, labels], index=['Number', 'Range']).T
        else:
            labels = r.astype(str).tolist()
            ohe_encoder = onehotencoder()
            ohe_encoder.fit(self.cut_labels)
            # cols = [f'Range_{self.cut_labels[i]}' for i in range(len(self.cut_labels))]         # for pd.DataFrame
            data = ohe_encoder.transform(labels)
            return data.toarray()
