import json
from collections import defaultdict
import pandas as pd
import pyecharts
import numpy as np
from scipy.stats import ks_2samp
import os

from .psi import Psi

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def get_score_data(s, key='score'):
    try:
        return json.loads(s)['data'][key]
    except:
        return np.nan


def parse_feature(s, model_id=None):
    return pd.Series(json.loads(s))


def k_stats(l):
    return {'median': l.median(), 'std': l.std(), 'mean': l.mean()}


def get_psi(df, hue='model_id', score_cols='model_record_response_data', benchmark_cols='test_f',
            pass_cols='is_grant', time_cols='model_record_created_at', score_func=get_score_data, score_keys='score'):
    """
    计算模型的psi
    :param df: dataframe
    :param hue: column names, like "model_id", which values represent different model.
    :param score_cols: column names, which values are json included score.
    :param benchmark_cols: column names, 线下benchmark集或者线上结果集的标识. {1: 线下benchmark集, 0: 线上结果集}
    :param pass_cols: column names, 是否放款标识，用来统计通过量。 {1：放款, 0: 不放款}
    :param time_cols: column names, 时间标识，用于按时间计算psi，比如按天计算psi
    :param score_func: 获取score的函数，支持自定义, 关键字和score_keys绑定。
                       score_keys参数考虑扔掉，如果需要换关键字直接通过自定义函数方式
    :return:
    """
    dic_psi = defaultdict(list)
    psi = Psi()
    for i in df[hue].unique():
        df_feature = df[df[hue] == i]
        df_feature['score'] = df_feature[score_cols].map(lambda s: score_func(s, key=score_keys))
        df_feature = df_feature.dropna(subset=['score'], axis=0)

        for t in df_feature[df_feature[benchmark_cols] == 0][time_cols].unique():
            psi.get_cutpoints(df_feature.score[df_feature[benchmark_cols] == 1])
            psi_value = round(
                float(psi.get_psi(
                    df_feature.score[(df_feature[benchmark_cols] == 0) & (df_feature[time_cols] == t)])),
                3)
            volume_value = psi.dist.actual_score.sum()
            pass_value = df[pass_cols][(df[benchmark_cols] == 0) & (df[time_cols] == t) & (df[hue] == i)].sum()
            pass_ratio = round(float(pass_value / volume_value), 3)
            dic_psi[hue + '_' + str(i)].append(
                {'date': t, 'psi': psi_value, 'volume': volume_value, 'pass': pass_value, 'pass_ratio': pass_ratio})
    return dic_psi


def get_feature_stats(df, hue='model_id', feature_cols='model_record_request_data', benchmark_cols='test_f',
                      time_cols='model_record_created_at', stats_func=k_stats, parse_func=parse_feature, cols=None):
    """
    计算特征ks值以及统计量
    :param df:
    :param hue:
    :param feature_cols:
    :param score_keys:
    :param benchmark_cols:
    :param pass_cols:
    :param time_cols:
    :param cols: 指定需要计算的column
    :param stats_func: 计算统计量函数，支持自定义
    :param parse_func: 解析json, 内部运算, 生成真实的feature并分列, 应该将对应模型内部特征处理逻辑抽象成一个函数传入此处;
                       如果是df中包含多个模型，分别有不同的处理逻辑，通过model_id参数，使用if then方式囊括所有处理方式
    :return:
    """
    dic_ks = {}
    dic_stats = {}
    for i in df[hue].unique():
        dic_ks[hue + '_' + str(i)] = defaultdict(list)
        dic_stats[hue + '_' + str(i)] = defaultdict(list)
        df_feature = df[df[hue] == i]
        df_fea = df_feature[feature_cols].apply(lambda s: parse_func(s, i))
        df_fea = df_fea.replace('', np.nan).convert_objects(convert_numeric=True)
        df_feature = pd.concat([df_feature, df_fea], axis=1)

        if cols is None:
            cols_f = df_fea.columns
        else:
            cols_f = [f for f in cols if f in df_fea.columns]

        for f in cols_f:
            try:
                te_v = k_stats(df_feature[f][df_feature[benchmark_cols] == 1])
                t_list = df_feature[df_feature[benchmark_cols] == 0][time_cols].unique()
                for t in t_list:
                    temp = df_feature[f][(df_feature[benchmark_cols] == 0) & (df_feature[time_cols] == t)]

                    ks = ks_2samp(df_feature[f][df_feature[benchmark_cols] == 1], temp)[0]
                    dic_ks[hue + '_' + str(i)][f].append(
                        {'date': t, 'ks': round(float(ks), 3), 'volume': temp.shape[0]})

                    tr_v = stats_func(temp)
                    dic_stats[hue + '_' + str(i)][f].append(
                        {'date': t, 'te_v': te_v, 'tr_v': tr_v, 'volume': temp.shape[0]})
            except Exception as e:
                print(f + ' can\'t compute stats values: ' + str(e))
    return dic_ks, dic_stats


def plot_monitor(dic_psi, dic_ks, dic_stats, cols=None, path='reportsource/model_monitor.html'):
    if not os.path.exists(os.path.dirname(os.path.abspath(path))):
        os.makedirs(os.path.dirname(os.path.abspath(path)))

    page = pyecharts.Page()

    for k in dic_psi.keys():
        temp = pd.read_json(json.dumps(dic_psi[k], cls=MyEncoder)).sort_values('date')
        temp = temp[temp.volume > 100]
        line = pyecharts.Line(k + ' PSI监控')
        line.add('psi', temp.date, temp['psi'], xaxis_rotate=45, is_datazoom_show=True, tooltip_tragger='axis',
                 datazoom_type='both', legend_pos='center', legend_orient='horizontal')

        line_pass = pyecharts.Line()
        line_pass.add('pass_ratio', temp.date, temp['pass_ratio'], xaxis_rotate=45, is_datazoom_show=True,
                      tooltip_tragger='axis', datazoom_type='both', legend_pos='center', legend_orient='horizontal')

        bar = pyecharts.Bar()
        bar.add('volume', temp.date, temp['volume'], xaxis_rotate=45, is_datazoom_show=True, tooltip_tragger='axis',
                datazoom_type='both', legend_pos='center', legend_orient='horizontal')

        overlap = pyecharts.Overlap()
        overlap.add(line)
        overlap.add(line_pass)
        overlap.add(bar, yaxis_index=1, is_add_yaxis=True)

        page.add(overlap)

    for k in dic_ks.keys():
        line = pyecharts.Line('%s features ks monitor' % k)
        for f in dic_ks[k].keys():
            temp = pd.read_json(json.dumps(dic_ks[k][f], cls=MyEncoder))
            temp = temp[temp['volume'] > 100]
            line.add(f, temp.date, temp.ks, xaxis_rotate=90, is_datazoom_show=True, tooltip_tragger='axis',
                     datazoom_type='both', legend_pos='right', legend_orient='vertical')

        page.add(line)

    for k in dic_stats.keys():
        if cols is None:
            cols_f = dic_stats[k].keys()
        else:
            cols_f = [f for f in cols if f in dic_stats[k].keys()]
        for f in cols_f:
            line = pyecharts.Line('%s features stats monitor: ' % k + f)

            temp = pd.DataFrame(dic_stats[k][f]).sort_values('date')
            stats_name = temp.iloc[0]['tr_v'].keys()
            for n in stats_name:
                temp_te_v = temp.te_v.apply(pd.Series)
                temp_tr_v = temp.tr_v.apply(pd.Series)

                line.add(n, temp.date, temp_te_v[n], is_datazoom_show=True, tooltip_tragger='axis',
                         datazoom_type='both', legend_pos='right', legend_orient='vertical')
                line.add(n + '_benchmark', temp.date, temp_tr_v[n], is_datazoom_show=True, tooltip_tragger='axis',
                         datazoom_type='both', legend_pos='right', legend_orient='vertical')

            page.add(line)
    page.render(path)
    return None


def model_monitor(df, hue='model_id', score_cols='model_record_response_data', benchmark_cols='test_f',
                  pass_cols='is_grant', time_cols='model_record_created_at',
                  score_func=get_score_data, score_keys='score', feature_cols='model_record_request_data',
                  stats_func=k_stats, parse_func=parse_feature, cols=None, path='reportsource/model_monitor.html'):
    dic_psi = get_psi(df, hue=hue, score_cols=score_cols, benchmark_cols=benchmark_cols, pass_cols=pass_cols,
                      time_cols=time_cols, score_func=score_func, score_keys=score_keys)
    dic_ks, dic_stats = get_feature_stats(df, hue=hue, feature_cols=feature_cols, benchmark_cols=benchmark_cols,
                                          time_cols=time_cols, stats_func=stats_func, parse_func=parse_func, cols=cols)
    plot_monitor(dic_psi, dic_ks, dic_stats, cols, path)
    return None

if __name__ == '__main__':
    df = pd.read_pickle('../test_data/model_monitor_test_data.pkl')
    model_monitor(df, path='../test_output/model_monitor.html')
