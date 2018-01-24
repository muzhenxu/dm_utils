from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

params_tree = {
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


# def xgb_model_evaluation(df, target, params='gbtree', split_size=[0.2, 0.1], random_state=7, early_stopping_rounds=100,
#                          num_rounds=50000, verbose_eval=True, pn_ratio=None):
#     if pn_ratio is None:
#         pn_ratio = np.sum(target == 0) / np.sum(target == 1)
#
#     if params == 'gbtree':
#         params = params_tree
#         params['scale_pos_weight'] = pn_ratio
#     elif params == 'gblinear':
#         params = params_linear
#         params['scale_pos_weight'] = pn_ratio
#
#     x_tr, x_va, y_tr, y_va = train_test_split(df, target, test_size=np.sum(split_size), random_state=random_state)
#     x_va, x_te, y_va, y_te = train_test_split(x_va, y_va, test_size=split_size[-1] / np.sum(split_size),
#                                               random_state=random_state)
#
#     indices = np.where(x_tr.sum(axis=0) == 0)[0]
#     if len(indices) > 0:
#         for i in indices:
#             try:
#                 x_tr[0, i] = 0.00000001
#             except:
#                 x_tr.iloc[0, i] = 0.00000001
#
#     indices = np.where(x_va.sum(axis=0) == 0)[0]
#     if len(indices) > 0:
#         for i in indices:
#             try:
#                 x_va[0, i] = 0.00000001
#             except:
#                 x_va.iloc[0, i] = 0.00000001
#
#     dtrain = xgb.DMatrix(x_tr, y_tr)
#     dvalid = xgb.DMatrix(x_va, y_va)
#     dtest = xgb.DMatrix(x_te, y_te)
#
#     watchlist = [(dtrain, 'train'), (dvalid, 'val')]
#     bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds,
#                     verbose_eval=verbose_eval)
#     y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration + 1)
#     try:
#         col_name = target.name
#     except:
#         col_name = 'y_true'
#     df_pred = pd.DataFrame({col_name: y_te, 'y_pred': y_pred})
#     print('test auc: ', roc_auc_score(y_te, y_pred))
#     return bst, df_pred


def xgb_model_evaluation(df, target, test=None, test_y=None, params='gbtree', n_folds=5, test_size=0.2, random_state=7,
                         early_stopping_rounds=100, num_rounds=50000, cv_verbose_eval=False, verbose_eval=True, pn_ratio=None):
    # try:
    #     col_name = target.name
    # except:
    #     col_name = 'y_true'
    col_name = 'y_true'

    df = df
    target = target

    if pn_ratio is None:
        pn_ratio = np.sum(target == 0) / np.sum(target == 1)

    if params == 'gbtree':
        params = params_tree
        params['scale_pos_weight'] = pn_ratio
    elif params == 'gblinear':
        params = params_linear
        params['scale_pos_weight'] = pn_ratio

    if (test is None) & (test_y is None):
        train, test, train_y, test_y = train_test_split(df, target, test_size=test_size,
                                                        random_state=random_state)
    else:
        train = df
        train_y = target
        test = test
        test_y = test_y

    pred_val = []
    y_val = []
    pred_te = np.zeros(len(test))

    dtest = xgb.DMatrix(test)

    if n_folds:
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

            pred_val.extend(list(bst.predict(dval)))
            y_val.extend(list(val_y))

            if test_size > 0:
                temp = bst.predict(dtest)
                pred_te += temp
                print('train auc: ', roc_auc_score(tra_y, bst.predict(xgb.DMatrix(tra))))
                print('val auc: ', roc_auc_score(val_y, bst.predict(xgb.DMatrix(val))))
                print('test auc: ', roc_auc_score(test_y, temp))

        print('cv val auc: ', roc_auc_score(y_val, pred_val))
        df_val = pd.DataFrame({col_name: y_val, 'y_pred': pred_val})

        pred_te /= n_folds
        print('cv test auc: ', roc_auc_score(test_y, pred_te))

        dtrain = xgb.DMatrix(train, train_y)
        bst = xgb.train(params, dtrain, num_boost_round=bst.best_iteration + 1, verbose_eval=verbose_eval)
    else:
        train, valid, train_y, valid_y = train_test_split(train, train_y, test_size=test_size,
                                                        random_state=random_state)
        dtrain = xgb.DMatrix(train, train_y)
        dvalid = xgb.DMatrix(valid, valid_y)
        dval = xgb.DMatrix(valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=verbose_eval)
        pred_val = bst.predict(dval)
        df_val = pd.DataFrame({col_name: valid_y, 'y_pred': pred_val})

    dtra = xgb.DMatrix(train)
    pred_tra = bst.predict(dtra)
    df_train = pd.DataFrame({col_name: train_y, 'y_pred': pred_tra})

    if test_size > 0:
        y_pred = bst.predict(dtest)

        df_pred = pd.DataFrame({col_name: test_y, 'y_pred': y_pred})
        print('test auc: ', roc_auc_score(test_y, y_pred))
    else:
        df_pred = None

    return bst, df_pred, df_val, df_train
