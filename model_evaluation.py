from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 多分类的问题
    # 'num_class': 2,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'lambda': 20,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'scale_pos_weight': 1,
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.01,  # 如同学习率
    'seed': 1000,
    # 'nthread': 7,  # cpu 线程数
    'eval_metric': 'auc'
}

params_tree = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 多分类的问题
    # 'num_class': 2,  # 类别数，与 multisoftmax 并用
    'gamma': 0.7,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 2,  # 构建树的深度，越大越容易过拟合
    'lambda': 20,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'scale_pos_weight': 1,
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.5,  # 如同学习率
    'seed': 1000,
    # 'nthread': 7,  # cpu 线程数
    'eval_metric': 'auc'
}

params_linear = {
    'booster': 'gblinear',
    'objective': 'binary:logistic',  # 多分类的问题
    # 'num_class': 2,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'lambda': 20,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'scale_pos_weight': 1,
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.01,  # 如同学习率
    'seed': 1000,
    # 'nthread': 7,  # cpu 线程数
    'eval_metric': 'auc'
}


def xgb_model_evaluation(df, target, test=None, test_y=None, params='gbtree', n_folds=5, test_size=0.2, random_state=7,
                         early_stopping_rounds=100, num_rounds=50000, cv_verbose_eval=False, verbose_eval=True,
                         oversample=False):
    # try:
    #     col_name = target.name
    # except:
    #     col_name = 'y_true'
    if (test_size == 0) & (n_folds is None):
        raise Exception("Error: test_size and n_folds can't both invalid.")

    col_name = 'y_true'
    best_iteration = 0

    if params == 'gbtree':
        params = params_tree
    elif params == 'gblinear':
        params = params_linear

    if oversample:
        pn_ratio = np.sum(target == 0) / np.sum(target == 1)
        params['scale_pos_weight'] = pn_ratio

    if (test is None) & (test_y is None):
        train, test, train_y, test_y = train_test_split(df, target, test_size=test_size,
                                                        random_state=random_state)
    else:
        train = df
        train_y = target
        test = test
        test_y = test_y
        test_size = 1

    dic_cv = []

    dtest = xgb.DMatrix(test)

    if n_folds:
        df_val = pd.DataFrame()
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for t_index, v_index in skf.split(train, train_y):
            tra, val = train.iloc[t_index, :], train.iloc[v_index, :]
            tra_y, val_y = train_y.iloc[t_index], train_y.iloc[v_index]

            dtrain = xgb.DMatrix(tra, tra_y)
            dvalid = xgb.DMatrix(val, val_y)
            dval = xgb.DMatrix(val)

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=cv_verbose_eval)

            print('best_iteration: ', bst.best_iteration)
            best_iteration += bst.best_iteration

            if test_size > 0:
                temp = bst.predict(dtest)
                dic_res = {'train_auc': roc_auc_score(tra_y, bst.predict(xgb.DMatrix(tra))),
                           'val_auc': roc_auc_score(val_y, bst.predict(dval)),
                           'test auc': roc_auc_score(test_y, temp)}
            else:
                dic_res = {'train_auc': roc_auc_score(tra_y, bst.predict(xgb.DMatrix(tra))),
                           'val_auc': roc_auc_score(val_y, bst.predict(dval))}

            val_df = pd.DataFrame({'y_true': val_y, 'y_pred': bst.predict(dval)})
            df_val = pd.concat([df_val, val_df], axis=0)

            print(dic_res)
            dic_cv.append(dic_res)

        df_cv = cmpt_cv(dic_cv)

    dtr = xgb.DMatrix(train)
    dtrain = xgb.DMatrix(train, train_y)

    if test_size == 0:
        dvalid = xgb.DMatrix(train, train_y)
        best_iteration = best_iteration // n_folds + 1
        early_stopping_rounds = None
    else:
        dvalid = xgb.DMatrix(test, test_y)
        try:
            best_iteration = best_iteration // n_folds + 1
            early_stopping_rounds = None
        except:
            best_iteration = 20000

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=best_iteration, evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval)

    if test_size > 0:
        pred_test = bst.predict(dtest)
        df_test = pd.DataFrame({col_name: test_y, 'y_pred': pred_test})
    else:
        df_test = df_val
    pred_train = bst.predict(dtr)
    df_train = pd.DataFrame({col_name: train_y, 'y_pred': pred_train})


    return bst, df_cv, df_test, df_train


def cmpt_cv(dic_cv):
    df_cv = pd.DataFrame(dic_cv)
    df_cv = df_cv.describe().loc[['mean', 'std', 'min', 'max']]
    return df_cv
