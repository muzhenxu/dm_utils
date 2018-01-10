from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

params_tree = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 多分类的问题
    # 'num_class': 2,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'scale_pos_weight': 1,
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,  # 如同学习率
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
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'scale_pos_weight': 1,
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,  # 如同学习率
    'seed': 1000,
    # 'nthread': 7,  # cpu 线程数
    'eval_metric': 'auc'
}


def xgb_model_evaluation(df, target, params='gbtree', split_size=[0.2, 0.1], random_state=7, early_stopping_rounds=100, num_rounds=50000, verbose_eval=True, pn_ratio=None):
    if pn_ratio is None:
        pn_ratio = np.sum(target==0) / np.sum(target==1)

    if params == 'gbtree':
        params = params_tree
        params['scale_pos_weight'] = pn_ratio
    elif params == 'gblinear':
        params = params_linear
        params['scale_pos_weight'] = pn_ratio

    x_tr, x_va, y_tr, y_va = train_test_split(df, target, test_size=np.sum(split_size), random_state=random_state)
    x_va, x_te, y_va, y_te = train_test_split(x_va, y_va, test_size=split_size[-1]/np.sum(split_size), random_state=random_state)

    indices = np.where(x_tr.sum(axis=0) == 0)[0]
    if len(indices) > 0:
        for i in indices:
            try:
                x_tr[0, i] = 0.00000001
            except:
                x_tr.iloc[0, i] = 0.00000001

    dtrain = xgb.DMatrix(x_tr, y_tr)
    dvalid = xgb.DMatrix(x_va, y_va)
    dtest = xgb.DMatrix(x_te, y_te)

    watchlist = [(dtrain, 'train'), (dvalid, 'val')]
    bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
    y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration+1)
    try:
        col_name = target.name
    except:
        col_name = 'y_true'
    df_pred = pd.DataFrame({col_name: y_te, 'y_pred': y_pred})
    print('test auc: ', roc_auc_score(y_te, y_pred))
    return bst, df_pred
