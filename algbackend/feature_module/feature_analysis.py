import sys

sys.path.append('.')
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.tree import DecisionTreeClassifier
import math
from collections import defaultdict
from scipy.stats import ks_2samp
from sklearn import metrics as mr
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandasql as ps

plt.rcParams['font.sans-serif'] = ['SimHei']  # 动态设置 字体
plt.rcParams['axes.unicode_minus'] = False
thres = '%.2f%%' % (0.05 * 100)


# 区分离散变量与连续变量
def choose_dis_con_variables(dataframe, label, number=20):
    x_cols = [x for x in dataframe.columns if x not in label]
    df_count = dataframe.loc[:, x_cols].apply(lambda x: x.nunique(), axis=0)
    continous_variables = df_count[df_count > number].index
    discrete_variables = df_count[df_count <= number].index
    return [continous_variables, discrete_variables]


##___________________________________特征选择___________________________________

# --------------------------------------------离散变量--------------------------------------------

def compute_overall_overdue(dataframe, label):
    all_count = dataframe.shape[0]
    is_yq_count = np.sum(dataframe[label] == 1)
    not_yq_count = np.sum(dataframe[label] == 0)
    overall_overduerate = is_yq_count / all_count
    return [all_count, is_yq_count, not_yq_count, overall_overduerate]


def compute_correlation(t):
    if t > 110:
        return 3
    elif t < 90:
        return 1
    else:
        return 2


def choose_discete_features(var, label, dataframe, thres):
    bad = np.sum(dataframe[label] == 1)
    good = np.sum(dataframe[label] == 0)
    all_count = dataframe.shape[0]
    overall_overduerate = bad / all_count
    var_df = ps.sqldf(
        "select (%s), sum((%s)) as '逾期个数',count(1) as '申贷个数' from dataframe group by (%s);" % (var, label, var),
        locals())
    var_df['正向指数'] = round((100 * var_df['逾期个数'] / var_df['申贷个数']) / overall_overduerate, 0).astype(int)
    var_df['申贷占比'] = var_df['申贷个数'] / all_count
    var_df['逾期率'] = var_df['逾期个数'] / var_df['申贷个数']
    var_df['申贷占比'] = var_df['申贷占比'].apply(lambda x: '%.3f%%' % (x * 100))
    var_df['逾期率'] = var_df['逾期率'].apply(lambda x: '%.3f%%' % (x * 100))
    var_df['相关性'] = 2
    for i in np.arange(var_df.shape[0]):
        if float(var_df.loc[var_df.index[i], '申贷占比'].split('%')[0]) > float(thres.split('%')[0]) and var_df.loc[
            var_df.index[i], '正向指数'] >= 110:
            var_df.loc[var_df.index[i], '相关性'] = 3
        elif float(var_df.loc[var_df.index[i], '申贷占比'].split('%')[0]) > float(thres.split('%')[0]) and var_df.loc[
            var_df.index[i], '正向指数'] <= 90:
            var_df.loc[var_df.index[i], '相关性'] = 1
        else:
            var_df.loc[var_df.index[i], '相关性'] = 2
    var_df['WOE'] = np.log(((var_df['逾期个数']) / bad) / ((var_df['申贷个数'] - var_df['逾期个数']) / good))
    var_df['iv'] = (var_df['逾期个数'] / bad - (var_df['申贷个数'] - var_df['逾期个数']) / good) * var_df['WOE']
    return var_df


def remark_color(row, var_name, writer, var_df):
    workbook = writer.book
    worksheet = writer.sheets[var_name]
    header_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '00BFFF',
        'border': 1,
        'bold': True})
    coverage_format = workbook.add_format({
        'fg_color': '#EE82EE',
        'border': 1
    })
    negative_format = workbook.add_format({
        'fg_color': '#FFFF00',
        'border': 1
    })
    positive_format = workbook.add_format({
        'fg_color': '#D7E4BC',
        'border': 1})
    all_format = workbook.add_format({
        'border': 1})
    for col_num, value in enumerate(var_df.columns.values):
        worksheet.write(row, col_num + 1, value, header_format)
    for i in np.arange(row + 1, row + var_df.shape[0] + 1):  # 行
        for j in np.arange(var_df.shape[1]):  # 列
            r = i - row - 1
            value = var_df.iloc[r, j]
            worksheet.write(i, j + 1, value, all_format)
            if var_df.columns[j] == '申贷占比' and float(value.split('%')[0]) > float(thres.split('%')[0]):
                worksheet.write(i, j + 1, value, coverage_format)
            if var_df.ix[r, '相关性'] == 1:
                worksheet.write(i, j + 1, value, negative_format)
            if var_df.ix[r, '相关性'] == 3:
                worksheet.write(i, j + 1, value, positive_format)


def write_Iv(row, var_name, writer, var_df):
    workbook = writer.book
    worksheet = writer.sheets[var_name]
    header_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#00BFFF',
        'border': 1,
        'bold': True})
    all_format = workbook.add_format({
        'border': 1})
    for col_num, value in enumerate(var_df.columns.values):
        worksheet.write(row, col_num + 1, value, header_format)
    for i in np.arange(row + 1, row + var_df.shape[0] + 1):  # 行
        for j in np.arange(var_df.shape[1]):
            r = i - row - 1
            value = var_df.iloc[r, j]
            worksheet.write(i, j + 1, value, all_format)


def discete_write_dataframe_to_excel(features, label, dataframe, thres, var_name, excel_name):
    chart_row = 0
    writer = pd.ExcelWriter(excel_name, options={'nan_inf_to_errors': True})
    df_temp = pd.concat([dataframe.loc[:, features], dataframe.loc[:, label]], axis=1)
    df_iv = pd.DataFrame(columns=['Var', 'IV'])
    for ind, feat in enumerate(features):
        temp = choose_discete_features(feat, label, df_temp, thres)
        df_tp = pd.DataFrame(np.array([feat, temp['iv'].sum()]).reshape(1, -1), columns=['Var', 'IV'])
        df_iv = pd.concat([df_tp, df_iv], axis=0)
        temp.to_excel(writer, sheet_name=var_name, startrow=chart_row, startcol=0)
        remark_color(chart_row, var_name, writer, temp)
        chart_row = chart_row + temp.shape[0] + 2
    df_iv.index = np.arange(df_iv.shape[0])
    df_iv.IV = df_iv.IV.astype(dtype=float)
    df_iv = df_iv.sort_values(by=['IV'], ascending=False)
    df_iv.to_excel(writer, sheet_name=var_name, startrow=chart_row, startcol=0)
    write_Iv(chart_row, var_name, writer, df_iv)
    writer.save()
    writer.close()


# --------------------------------------------连续变量--------------------------------------------
def split_training_validation(df_x, df_y, ratio=0.2):
    for seed in np.arange(10000):
        x_train, x_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size=ratio, random_state=seed)
        df_train = pd.concat([x_train, Y_train], axis=1)
        df_test = pd.concat([x_test, Y_test], axis=1)
        train_overdue = round(np.sum(Y_train == 1) / Y_train.shape[0], 3)
        test_overdue = round(np.sum(Y_test == 1) / Y_test.shape[0], 3)
        if np.abs(train_overdue - test_overdue) <= 0.001:
            print('The seed is %s' % seed)
            temp = pd.DataFrame(np.array([train_overdue, test_overdue]).reshape(-1, 1), index=['train', 'validation'],
                                columns=['overdue_rate'])
            break
    return [df_train, df_test, temp]


def compute_groupceil(xx, low_value, high_value, number_of_bins, range0, null_value):
    if xx == null_value:
        bins = 0
    else:
        if xx <= low_value:
            bins = 1
        elif xx >= high_value:
            bins = number_of_bins + 2
        else:
            bins = np.ceil((xx - low_value) / range0) + 1
    return bins


def compute_continous_variables(dataframe, null_value, feat, overall_overduerate, low_quantile=0.01, high_quantile=0.99,
                                number_of_bins=10, parts='training'):
    dataframes = dataframe.copy()
    xx = dataframes.loc[:, feat].copy()
    x_notnull = xx[xx != null_value]
    quantile_values = x_notnull.quantile([low_quantile, high_quantile])
    range0 = (quantile_values[high_quantile] - quantile_values[low_quantile]) / number_of_bins  # 每一段的长度
    low_value = quantile_values[low_quantile]
    high_value = quantile_values[high_quantile]
    dataframes['groupceil'] = dataframes[feat].apply(
        lambda t: compute_groupceil(t, low_value, high_value, number_of_bins, range0, null_value))
    df_sum = ps.sqldf(
        "select groupceil, count(1) as count, sum(%s) as yq_count, avg(%s) as ceil_value, avg(%s) as avg_is_yq from dataframes group by groupceil;" % (
            label, feat, label), locals())
    df_sum['Var'] = [feat] * df_sum.shape[0]
    df_sum['partion'] = ['%s' % parts] * df_sum.shape[0]
    df_sum['group_total'] = [dataframe.shape[0]] * df_sum.shape[0]
    df_sum['dist_pct'] = df_sum['count'] / df_sum['group_total']
    b_var = df_sum.pop('Var')
    df_sum.insert(0, 'Var', b_var)
    b_p = df_sum.pop('partion')
    df_sum.insert(1, 'partion', b_p)
    all_bad = (dataframe[label] == 1).sum()
    all_good = (dataframe[label] == 0).sum()
    all_ratio = all_good / all_bad
    for i in np.arange(df_sum.shape[0]):
        good = df_sum.loc[df_sum.index[i], 'count'] - df_sum.loc[df_sum.index[i], 'yq_count']
        bad = df_sum.loc[df_sum.index[i], 'yq_count']
        df_sum.loc[df_sum.index[i], 'woe'] = np.log(all_ratio * ((bad + 1) / (1 + good)))
        df_sum.loc[df_sum.index[i], 'iv'] = (bad / all_bad - good / all_good) * df_sum.loc[df_sum.index[i], 'woe']
    df_sum['IV_total'] = [df_sum.iv.sum(skipna=False)] * df_sum.shape[0]
    df_sum['profile'] = round((100 * df_sum['yq_count'] / df_sum['count']) / overall_overduerate, 0).astype(int)
    return df_sum


def remark_color_continous(row, var_name, writer, var_df, null_value):
    workbook = writer.book
    worksheet = writer.sheets[var_name]
    header_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#00BFFF',
        'border': 1,
        'bold': True})
    all_format = workbook.add_format({
        'border': 1})
    null_format = workbook.add_format({
        'fg_color': '#D7E4BC',
        'border': 1})
    fill_format = workbook.add_format({
        'fg_color': '#EE82EE',
        'border': 1})
    temp = var_df.loc[var_df.partion == 'training', ['ceil_value', 'avg_is_yq']]
    if null_value == temp.ix[0, 'ceil_value']:
        rate = temp.ix[0, 'avg_is_yq']
        temp['diff'] = temp['avg_is_yq'].apply(lambda x: x - rate)
        temp['diff'] = np.abs(temp['diff'])
        m = min(temp.loc[temp['diff'] != 0, 'diff'])
        index_min = temp.loc[temp['diff'] == m, 'diff'].index
    for col_num, value in enumerate(var_df.columns.values):
        worksheet.write(row, col_num + 1, value, header_format)
    for i in np.arange(row + 1, row + var_df.shape[0] + 1):  # 行
        for j in np.arange(var_df.shape[1]):
            r = i - row - 1
            value = var_df.iloc[r, j]
            worksheet.write(i, j + 1, value, all_format)
            if var_df.ix[r, 'ceil_value'] == null_value:
                worksheet.write(i, j + 1, value, null_format)
            if null_value == temp.ix[0, 'ceil_value'] and r in index_min:
                worksheet.write(i, j + 1, value, fill_format)


def continous_plot_to_excel(writer, var_name, feat, amin, ymin, y_name, row, plas_train, plas_test, train_shape,
                            val_shape, insert_index, col_index):
    workbook = writer.book
    worksheet = writer.sheets[var_name]
    chart = workbook.add_chart({'type': 'scatter'})
    chart.add_series({
        'name': [var_name, row, 2],
        'categories': [var_name, row + plas_train, 6, row + train_shape - 1, 6],
        'values': [var_name, row + plas_train, col_index, row + train_shape - 1, col_index],
        'line': {'color': 'red', 'width': 1.25},
        'marker': {'type': 'diamond'}
    })
    chart.add_series({
        'name': [var_name, row + train_shape, 2],
        'categories': [var_name, row + train_shape + plas_test, 6, row + val_shape + train_shape - 1, 6],
        'values': [var_name, row + train_shape + plas_test, col_index, row + val_shape + train_shape - 1, col_index],
        'line': {'color': 'green', 'width': 1.25},
        'marker': {'type': 'square'}
    })
    chart.set_x_axis({'name': '%s' % feat,
                      'min': amin,
                      })
    chart.set_y_axis({'name': y_name,
                      'min': ymin})
    chart.set_size({'width': 500, 'height': 300})
    chart.set_legend({'position': 'bottom'})
    chart.set_chartarea({
        'border': {'color': 'black'}})
    worksheet.insert_chart('%s%s' % (insert_index, row + 1), chart, {'x_offset': 25, 'y_offset': 10})


def min_value(x):
    if x < 0:
        xt = np.abs(x)
        v = float(str(xt).split('.')[0] + '.' + str(xt).split('.')[1][0]) + 0.1
        values = -1 * v
    elif x > 0:
        values = float(str(x).split('.')[0] + '.' + str(x).split('.')[1][0])
    else:
        values = 0
    return values


def continous_write_dataframe_to_excel(features, label, dataframe, var_name, excel_name, null_value,
                                       overall_overduerate):
    chart_row = 0
    writer = pd.ExcelWriter(excel_name, options={'nan_inf_to_errors': True})
    df_iv = pd.DataFrame(columns=['Var', 'IV_total'])
    df_train, df_test, df_overdue = split_training_validation(dataframe.loc[:, features], dataframe.loc[:, label],
                                                              ratio=0.2)
    for ind, feat in enumerate(features):
        print(feat)
        df_sum_train = compute_continous_variables(df_train, null_value, feat, overall_overduerate, low_quantile=0.01,
                                                   high_quantile=0.99, number_of_bins=10, parts='training')
        df_sum_val = compute_continous_variables(df_test, null_value, feat, overall_overduerate, low_quantile=0.01,
                                                 high_quantile=0.99, number_of_bins=10, parts='validation')
        df_temp = df_sum_train.loc[:, ['Var', 'IV_total']][0:1]
        df_iv = pd.concat([df_iv, df_temp], axis=0)
        df_sum = pd.concat([df_sum_train, df_sum_val], axis=0)
        df_sum.index = np.arange(df_sum.shape[0])
        df_sum.to_excel(writer, sheet_name=var_name, startrow=chart_row, startcol=0)
        remark_color_continous(chart_row, var_name, writer, df_sum, null_value)
        aamin = round(min(df_sum.loc[df_sum['ceil_value'] != null_value, 'ceil_value']), 4)
        yymin = round(min(df_sum.loc[df_sum['ceil_value'] != null_value, 'avg_is_yq']), 4)
        wwmin = round(min(df_sum.loc[df_sum['ceil_value'] != null_value, 'woe']), 4)
        ddmin = round(min(df_sum.loc[df_sum['ceil_value'] != null_value, 'dist_pct']), 4)
        ymin = min_value(yymin)
        dmin = min_value(ddmin)
        wmin = min_value(wwmin)
        amin = np.floor(aamin)
        train_shape = df_sum_train.shape[0]
        val_shape = df_sum_val.shape[0]
        if null_value in df_sum_train['ceil_value'].values and null_value in df_sum_val['ceil_value'].values:
            continous_plot_to_excel(writer, var_name, feat, amin, wmin, 'Woe', chart_row + 1, 1, 1, train_shape,
                                    val_shape, 'O', 10)
            continous_plot_to_excel(writer, var_name, feat, amin, ymin, 'Avg_is_yq', chart_row + 1, 1, 1, train_shape,
                                    val_shape, 'X', 7)
            continous_plot_to_excel(writer, var_name, feat, amin, dmin, 'Dist_pct', chart_row + 1, 1, 1, train_shape,
                                    val_shape, 'AG', 9)
        elif null_value in df_sum_train['ceil_value'].values and null_value not in df_sum_val['ceil_value'].values:
            continous_plot_to_excel(writer, var_name, feat, amin, wmin, 'Woe', chart_row + 1, 1, 0, train_shape,
                                    val_shape, 'O', 10)
            continous_plot_to_excel(writer, var_name, feat, amin, ymin, 'Avg_is_yq', chart_row + 1, 1, 0, train_shape,
                                    val_shape, 'X', 7)
            continous_plot_to_excel(writer, var_name, feat, amin, dmin, 'Dist_pct', chart_row + 1, 1, 0, train_shape,
                                    val_shape, 'AG', 9)
        elif null_value not in df_sum_train['ceil_value'].values and null_value not in df_sum_val['ceil_value'].values:
            continous_plot_to_excel(writer, var_name, feat, amin, wmin, 'Woe', chart_row + 1, 0, 0, train_shape,
                                    val_shape, 'O', 10)
            continous_plot_to_excel(writer, var_name, feat, amin, ymin, 'Avg_is_yq', chart_row + 1, 0, 0, train_shape,
                                    val_shape, 'X', 7)
            continous_plot_to_excel(writer, var_name, feat, amin, dmin, 'Dist_pct', chart_row + 1, 0, 0, train_shape,
                                    val_shape, 'AG', 9)
        else:
            continous_plot_to_excel(writer, var_name, feat, amin, wmin, 'Woe', chart_row + 1, 0, 1, train_shape,
                                    val_shape, 'O', 10)
            continous_plot_to_excel(writer, var_name, feat, amin, ymin, 'Avg_is_yq', chart_row + 1, 0, 1, train_shape,
                                    val_shape, 'X', 7)
            continous_plot_to_excel(writer, var_name, feat, amin, dmin, 'Dist_pct', chart_row + 1, 0, 1, train_shape,
                                    val_shape, 'AG', 9)
        chart_row = chart_row + df_sum.shape[0] + 2
    df_iv.index = np.arange(df_iv.shape[0])
    df_iv.IV_total = df_iv.IV_total.astype(dtype=float)
    df_iv = df_iv.sort_values(by=['IV_total'], ascending=False)
    df_iv.to_excel(writer, sheet_name=var_name, startrow=chart_row, startcol=0)
    write_Iv(chart_row, var_name, writer, df_iv)
    writer.save()
    writer.close()


if __name__ == '__main__':
    df = pd.read_pickle('...')

    continous_variables, discrete_variables = choose_dis_con_variables(df, label, number=20)

    label = 'is_yq'

    var_name = 'discrete_variable'
    excel_name = 'discrete.xlsx'
    features = discrete_variables
    all_count, is_yq_count, not_yq_count, overall_overduerate = compute_overall_overdue(df, 'is_yq')
    discete_write_dataframe_to_excel(features, label, df, thres, var_name, excel_name)

    # 缺失值赋值
    null_value = -10000

    var_name = 'continous_variable'
    excel_name = 'xu_continous.xlsx'
    features = continous_variables

    continous_write_dataframe_to_excel(features, label, df, var_name, excel_name, null_value, overall_overduerate)


