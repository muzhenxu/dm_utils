import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost as xgb


def Xgboost_Pred(bst, data, num_class):
    ddata = xgb.DMatrix(data)
    if num_class == 2:
        xgb_pred = bst.predict(ddata)
        xgb_pred = np.vstack([1 - xgb_pred, xgb_pred]).T
    else:
        xgb_pred = bst.predict(ddata)
    return xgb_pred


def balance_cv(train, test, valid, n_folds, col, col_cv_label_proba, label, clf, random_state, train_remain=None,
               train_r_all=None, param=None, num_round=None, l=1):
    kf = StratifiedKFold(train[label], n_folds=n_folds, shuffle=True, random_state=random_state)
    for t_index, v_index in kf:
        tra, val = train.iloc[t_index], train.iloc[v_index]
        if param is None:
            clf.fit(tra[col], tra[label])
            test_label = pd.DataFrame(clf.predict_proba(test[col]) / (n_folds * l), columns=col_cv_label_proba,
                                      index=test.index)
            valid[col_cv_label_proba] = valid[col_cv_label_proba].add(test_label, fill_value=0)
            val_label = pd.DataFrame(clf.predict_proba(val[col]) / (n_folds * (l - 1) + 1), columns=col_cv_label_proba,
                                     index=val.index)
            valid[col_cv_label_proba] = valid[col_cv_label_proba].add(val_label, fill_value=0)
            if train_remain is not None:
                train_remain_label = pd.DataFrame(clf.predict_proba(train_remain[col]) / (n_folds * (l - 1) + 1),
                                                  columns=col_cv_label_proba, index=train_remain.index)
                valid[col_cv_label_proba] = valid[col_cv_label_proba].add(train_remain_label, fill_value=0)
            if train_r_all is not None:
                train_r_all_label = pd.DataFrame(clf.predict_proba(train_r_all[col]) / (n_folds * l),
                                                 columns=col_cv_label_proba, index=train_r_all.index)
                valid[col_cv_label_proba] = valid[col_cv_label_proba].add(train_r_all_label, fill_value=0)
        else:
            dtra = xgb.DMatrix(tra[col], tra[label])
            dval = xgb.DMatrix(val[col], val[label])
            watchlist = [(dtra, 'train'), (dval, 'eval')]
            num_class = len(col_cv_label_proba)
            bst = xgb.train(param, dtra, num_round, watchlist)
            xgb_pred = Xgboost_Pred(bst, val[col], num_class)
            val_label = pd.DataFrame(xgb_pred / (n_folds * (l - 1) + 1), index=val.index, columns=col_cv_label_proba)
            valid[col_cv_label_proba] = valid[col_cv_label_proba].add(val_label, fill_value=0)
            xgb_pred = Xgboost_Pred(bst, test[col], num_class)
            test_label = pd.DataFrame(xgb_pred / (n_folds * l), index=test.index, columns=col_cv_label_proba)
            valid[col_cv_label_proba] = valid[col_cv_label_proba].add(test_label, fill_value=0)
            if train_remain is not None:
                xgb_pred = Xgboost_Pred(bst, train_remain[col], num_class)
                train_remain_label = pd.DataFrame(xgb_pred / (n_folds * (l - 1) + 1), index=train_remain.index,
                                                  columns=col_cv_label_proba)
                valid[col_cv_label_proba] = valid[col_cv_label_proba].add(train_remain_label, fill_value=0)
            if train_r_all is not None:
                xgb_pred = Xgboost_Pred(bst, train_r_all[col], num_class)
                train_r_all_label = pd.DataFrame(xgb_pred / (n_folds * l), columns=col_cv_label_proba,
                                                 index=train_r_all.index)
                valid[col_cv_label_proba] = valid[col_cv_label_proba].add(train_r_all_label, fill_value=0)
    return valid[col_cv_label_proba]


def imbalance_cv(train, test, valid, n_folds, col, col_cv_label_proba, majority_class, label, clf, random_state, param,
                 num_round, l):
    train_n = train[train[label] == majority_class]
    train_p = train[train[label] != majority_class]
    pred = None
    kf = KFold(train_n.shape[0], n_folds=l, shuffle=True, random_state=random_state)
    for r_index, n_index in kf:
        valid[col_cv_label_proba] = 0
        train_new = pd.concat([train_n.iloc[n_index], train_p], axis=0).sample(frac=1, random_state=random_state)
        train_remain = train_n.iloc[r_index]
        if pred is None:
            pred = balance_cv(train_new, test, valid, n_folds, col, col_cv_label_proba, label, clf, random_state,
                              train_remain, train_r_all=None, param=param, num_round=num_round, l=l)
        else:
            pred += balance_cv(train_new, test, valid, n_folds, col, col_cv_label_proba, label, clf, random_state,
                               train_remain, train_r_all=None, param=param, num_round=num_round, l=l)
    pred.loc[train_p.index] = pred.loc[train_p.index] * (n_folds * (l - 1) + 1) / l
    return pred


def serious_imbalance_cv(train, test, valid, n_folds, col, col_cv_label_proba, majority_class, label, random_state, clf,
                         param, num_round, l):
    train_n_all = train[train[label] == majority_class]
    train_p = train[train[label] != majority_class]
    n = train_p.shape[0] * l
    train_n = train_n_all.sample(n=n, random_state=random_state)
    train_r_all = train_n_all.drop(train_n.index)
    pred = None
    kf = KFold(train_n.shape[0], n_folds=l, shuffle=True, random_state=random_state)
    for r_index, n_index in kf:
        valid[col_cv_label_proba] = 0
        train_new = pd.concat([train_n.iloc[n_index], train_p], axis=0).sample(frac=1, random_state=random_state)
        train_remain = train_n.iloc[r_index]
        if pred is None:
            pred = balance_cv(train_new, test, valid, n_folds, col, col_cv_label_proba, label, clf, random_state,
                              train_remain, train_r_all, param, num_round, l)
        else:
            pred += balance_cv(train_new, test, valid, n_folds, col, col_cv_label_proba, label, clf, random_state,
                               train_remain, train_r_all, param, num_round, l)
    pred.loc[train_p.index] = pred.loc[train_p.index] * (n_folds * (l - 1) + 1) / l
    return pred


class cross_validation_class(object):
    ##this calss is only appropriate the imbalance dataset which has two labels or\
    # has more labels but have only majority class which is more over others,and the rest are balance.
    ##npr is the number of negative samples/positive
    def __init__(self, label, n_folds, id_col, npr=1.0, random_state=7):
        self.label = label
        self.n_folds = n_folds
        self.id_col = id_col
        self.id_col.append(self.label)
        self.random_state = random_state
        self.npr = float(npr)
        if self.npr < 1:
            return 'npr must equal 1.0 or larger than 1.0!'

    def split(self, df):
        self.train = df[~df[self.label].isnull()]
        self.test = df[df[self.label].isnull()]
        self.valid = df[[self.label]]
        label_freq = self.train[self.label].value_counts().sort_index()
        self.majority_class = label_freq.argmax()
        self.col = df.columns.difference(self.id_col)
        self.l = (label_freq.max() * 1.0 / label_freq.sort_values().iloc[:-1].mean()) / self.npr

        for i in label_freq.index:
            self.valid[i] = 0
        self.col_cv_label_proba = self.valid.columns[1:]

        if label_freq.shape[0] != 2:
            print('--------------------Warning: the number of classes is %s!--------------------' % label_freq.shape[0])

        if self.l < 0.8:
            if self.l * self.npr < 0.8:
                return 'positive samples is more larger than negative samples!the ture npr is %s' % self.l * self.npr
            else:
                return 'negative samples is not enough to afford the npr %s,the ture npr is %s' % (
                self.npr, self.l * self.npr)
        else:
            self.l = round(self.l)

        return self

    def pred_proba(self, clf=None, l=None, col=None, param=None, num_round=None):
        if col is not None:
            self_col = col
        else:
            self_col = self.col
        if l is not None:
            self_l = l
        else:
            self_l = self.l

        if self_l == 1:
            return balance_cv(self.train, self.test, self.valid, self.n_folds, self_col, self.col_cv_label_proba,
                              self.label, clf, self.random_state, train_remain=None, train_r_all=None, param=param,
                              num_round=num_round, l=1)
        elif l is not None:
            return serious_imbalance_cv(self.train, self.test, self.valid, self.n_folds, self_col,
                                        self.col_cv_label_proba, self.majority_class, self.label, self.random_state,
                                        clf, param, num_round, l=l)
        else:
            return imbalance_cv(self.train, self.test, self.valid, self.n_folds, self_col, self.col_cv_label_proba,
                                self.majority_class, self.label, clf, self.random_state, param, num_round, self_l)

    # this function is used to output the label not the proba.
    def predict(self, clf=None, l=None, col=None, param=None, num_round=None):
        proba = self.pred_proba(clf, l, col, param, num_round)
        return proba.apply(lambda row: row.argmax(), axis=1)
