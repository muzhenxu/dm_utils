from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier

def model_cv_trainer(df, target, test=None, test_y=None, clf='lr', params=None, n_folds=5, test_size=0.2, random_state=7):
    if clf == 'lr':
        clf = LogisticRegression(**params)
    elif clf == 'xgb':
        clf = XGBClassifier(**params)
    else:
        raise Exception(f'clf {clf} is not defined!')

    if (test_size == 0) & (n_folds is None):
        raise Exception("Error: test_size and n_folds can't both invalid.")

    if (test is None) & (test_y is None):
        train, test, train_y, test_y = train_test_split(df, target, test_size=test_size,
                                                        random_state=random_state)
    else:
        train = df
        train_y = target
        test = test
        test_y = test_y
        test_size = 1

    col_name = 'y_true'

    dic_cv = []

    if n_folds:
        df_val = pd.DataFrame()
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for t_index, v_index in skf.split(train, train_y):
            tra, val = train.iloc[t_index, :], train.iloc[v_index, :]
            tra_y, val_y = train_y.iloc[t_index], train_y.iloc[v_index]

            clf.fit(tra, tra_y)
            temp_tra = clf.predict(tra)
            temp_val = clf.predict(val)

            if test_size > 0:
                temp_test = clf.predict(test)
                dic_res = {'train_auc': roc_auc_score(tra_y, temp_tra),
                           'val_auc': roc_auc_score(val_y, temp_val),
                           'test auc': roc_auc_score(test_y, temp_test)}
            else:
                dic_res = {'train_auc': roc_auc_score(tra_y, temp_tra),
                           'val_auc': roc_auc_score(val_y, temp_val)}

            val_df = pd.DataFrame({'y_true': val_y, 'y_pred': temp_val})
            df_val = pd.concat([df_val, val_df], axis=0)

            print(dic_res)
            dic_cv.append(dic_res)

        df_cv = cmpt_cv(dic_cv)
    else:
        df_cv = None

    clf.fit(train, train_y)

    if test_size > 0:
        pred_test = clf.predict(test)
        df_test = pd.DataFrame({col_name: test_y, 'y_pred': pred_test})
    else:
        df_test = df_val
    pred_train = clf.predict(train)
    df_train = pd.DataFrame({col_name: train_y, 'y_pred': pred_train})

    return clf, df_cv, df_test, df_train


def cmpt_cv(dic_cv):
    df_cv = pd.DataFrame(dic_cv)
    df_cv = df_cv.describe().loc[['mean', 'std', 'min', 'max']]
    return df_cv
