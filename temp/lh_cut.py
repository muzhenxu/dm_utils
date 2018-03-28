import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np
import pandas as pd
import math
import random


def auc_values(fpr, tpr):
    l = len(fpr)
    auc = 0
    for i in range(1, l):
        auc = auc + tpr[i] * (fpr[i] - fpr[i - 1])
    return auc


def basic_model_info(score, y, pos_label=1):
    """
    caculate thresholds,tpr,fpr and other fature_label like score value with label value
    returns :
    above_threshold_is_pos,thresholds,tpr,fpr,accuracy,recall,precision
    """
    if pos_label == 1:
        neg_label = 0  # make sure neg label is not same as the pos label value
    else:
        neg_label = 1

    if len(score) != len(y):
        raise Exception("y length %d not match score length %d" % (len(y), len(score)))
    if pos_label not in y:
        raise Exception("positive_label %d not in list y" % (pos_label))

    ix = np.argsort(score)
    half_len = math.ceil(len(score) / 2)
    y_reorder = y[ix]
    head_half_y = y_reorder[:half_len]
    tail_half_y = y_reorder[half_len:]
    score_high_more_positive_label = True
    if (head_half_y == pos_label).sum() > (tail_half_y == pos_label).sum():
        score_high_more_positive_label = False

    # print (score_high_more_positive_label)
    if score_high_more_positive_label:
        ix = ix[::-1]
    score = score[ix]
    y_reorder = y[ix]

    # print ((y_reorder[:half_len]==pos_label).sum()-(y_reorder[half_len:]==pos_label).sum())
    positive_count = (y == pos_label).sum()
    negtive_count = (y != pos_label).sum()

    predict_pos = 0
    thresholds = []
    tpr = []
    fpr = []
    accuracy = []  # 准确率
    recall = []  # 召回率
    precision = []  # 精确率
    tpr_tmp = 0
    fpr_tmp = 0
    true_pos = 0  # 真正
    false_pos = 0  # 假正
    false_neg = 0  # 假负
    true_neg = 0  # 真负
    st = dt.now()
    for i in range(len(score)):
        # predict_pos = [pos_label]*(i+1)
        # predict_neg = [neg_label]*(len(score)-(i+1))
        # predict_pos.extend(predict_neg)
        # predict_label_list = np.array(predict_pos)

        true_pos = y_reorder[:i + 1].sum()  # ((predict_label_list==pos_label)[(y==pos_label)]).sum()
        false_pos = (y_reorder[:i + 1] != pos_label).sum()  # ((predict_label_list==pos_label)[(y!=pos_label)]).sum()
        false_neg = y_reorder[i + 1:].sum()  # ((predict_label_list!=pos_label)[(y==pos_label)]).sum()
        true_neg = (y_reorder[i + 1:] != pos_label).sum()  # ((predict_label_list!=pos_label)[(y!=pos_label)]).sum()

        accuracy_tmp = (true_pos + true_neg) / (true_pos + false_pos + false_neg + true_neg)
        accuracy.append(accuracy_tmp)

        precision_tmp = true_pos / (true_pos + false_pos)
        precision.append(precision_tmp)

        recall_tmp = true_pos / (true_pos + false_neg)
        recall.append(recall_tmp)

        thresholds.append(score[i])
        if y_reorder[i] == pos_label:
            tpr_tmp = tpr_tmp + (1 / positive_count)
        else:
            fpr_tmp = fpr_tmp + (1 / negtive_count)
        tpr.append(tpr_tmp)
        fpr.append(fpr_tmp)
    ed = dt.now()
    # print ("basic_model_info while method  cosuming : %f seconds"%((ed-st).total_seconds()))
    above_threshold_is_pos = score_high_more_positive_label
    return above_threshold_is_pos, np.array(thresholds), np.array(tpr), np.array(fpr), np.array(accuracy), np.array(
        recall), np.array(precision)


def plot_data(x, y, name, output_dir=None):
    """用来 绘制 x,y 函数


    """
    fig = plt.figure(figsize=[15, 10])
    df_opd = pd.DataFrame([y], columns=x)
    xra = np.arange(len(x))
    plt.bar(xra, y, alpha=0.5, tick_label=x, label="count")
    if "int" in str(type(y[0])):
        for a, b in zip(xra, y):
            plt.text(a, b * 1.02, '%d' % b, ha='center')
    else:
        for a, b in zip(xra, y):
            plt.text(a, b * 1.02, '%.2f' % b, ha='center')
    df_opd.iloc[0, :].plot(linestyle='dashdot', marker='o')
    plt.legend()
    plt.title(name + '  distribution')
    plt.ylabel('num of group')
    plt.xticks(rotation=90)
    if output_dir is not None:
        plt.savefig(output_dir + name + "_distribution.jpg")
    plt.show()


def binary_search_sum(arr, start, i_start, i_end, numval):
    """
    传入排序后每个组的 个数 列表（numpy.array)
    arr:排序后列表 每个元素的个数
    start:开始元素位置（计算sum)
    i_start:移动范围开始指针
    i_end:移动范围终结指针
    numval: sum 值得最小个数

    """
    if arr[start:i_end].sum() <= numval:
        return len(arr)
    else:
        curosr = math.ceil((i_start + i_end) / 2)
        if arr[start:curosr].sum() == numval:
            #             print ("p1 %d , %d" % (i_start,i_end))
            return curosr
        elif arr[start:curosr].sum() < numval:
            #             print ("p2 %d , %d" % (i_start,i_end))
            return binary_search_sum(arr, start, curosr, i_end, numval)
        else:
            #             print ("p3 %d , %d" % (i_start,i_end))
            if arr[start:curosr - 1].sum() < numval:
                return curosr
            elif arr[start:curosr - 1].sum() == numval:
                return curosr - 1
            else:
                return binary_search_sum(arr, start, i_start, curosr - 1, numval)


def percent_str(x):
    if np.isnan(x):
        return x
    else:
        return "%.2f" % (x * 100) + "%"


def reorder_df(df, x_column):
    df.sort_values(x_column, inplace=True)


def optimized_split_df(df, feature_column_name, bins=10, as_type=None):
    if as_type != None:
        df.feature_column_name = df.feature_column_name.astype(as_type)
        # df.loc[not_null,feature_column_name]=df[not_null][feature_column_name].astype(as_type)
    # print (feature_column_name)
    not_null = ~df[feature_column_name].isnull()
    df_n = df[~not_null]  # 值空集合
    df_nn = df[not_null]  # 值非空集合
    reorder_df(df_nn, feature_column_name)
    kind_val = list(df_nn.groupby(feature_column_name).size().index.values)
    kind_size = df_nn.groupby(feature_column_name).size().values

    kind_size_unique = np.unique(kind_size)
    kind_size_unique.sort()
    # kind_size_unique=kind_size_unique[::-1]
    max_bin = len(kind_val)

    if max_bin < bins:
        print("branch small")
        print("warning: feature column only has %d different kind val not enough for bins %d." % (max_bin, bins))
        return inner_optimized_split_method(df_n, df_nn, kind_val, kind_size, feature_column_name, kind_size.min())
    elif max_bin == bins or bins == -1:
        print("branch equal")
        return inner_optimized_split_method(df_n, df_nn, kind_val, kind_size, feature_column_name, kind_size.min())
    else:
        print("branch 3")
        bins_tmp = bins
        min_num = math.ceil(len(df_nn) / bins_tmp)
        dfa, group_kind_vals_list, contain_null = inner_optimized_split_method(df_n, df_nn, kind_val, kind_size,
                                                                               feature_column_name, min_num)
        bins_tmp = bins_tmp + 1
        min_num_l = math.ceil(len(df_nn) / bins_tmp + 1)
        dfa_l, group_kind_vals_list_l, contain_null_l = inner_optimized_split_method(df_n, df_nn, kind_val, kind_size,
                                                                                     feature_column_name, min_num_l)
        while len(group_kind_vals_list_l) <= bins:
            dfa, group_kind_vals_list, contain_null = dfa_l, group_kind_vals_list_l, contain_null_l  # last
            bins_tmp = bins_tmp + 1
            min_num_l = math.ceil(len(df_nn) / bins_tmp)
            dfa_l, group_kind_vals_list_l, contain_null_l = inner_optimized_split_method(df_n, df_nn, kind_val,
                                                                                         kind_size, feature_column_name,
                                                                                         min_num_l)
            print("len:%d bins_tmp :%d" % (len(group_kind_vals_list_l), bins_tmp))
        return dfa, group_kind_vals_list, contain_null


def inner_optimized_split_method(df_n, df_nn, kind_val, kind_size, feature_column_name, min_num):
    st = dt.now()
    dfa = []
    group_kind_vals_list = []
    contain_null = False
    if len(df_n) != 0:
        dfa.append(df_n)
        contain_null = True
        group_kind_vals_list.append([df_n[feature_column_name].values[0]])
    end = 0
    while end != len(kind_val):
        start = end
        end = binary_search_sum(kind_size, start, start, len(kind_size), min_num)
        group_kind_vals = kind_val[start:end]
        group_kind_vals_list.append(group_kind_vals)
        #         print (type(group_kind_vals[0]))
        #         print (type(df_nn[feature_column_name].values[0]))
        dftmp = df_nn[df_nn[feature_column_name].isin(group_kind_vals, )]
        #         print ("kind_size length:"+str(len(kind_size))+"kind_val length: "+str(len(kind_val))+" end:"+str(end))
        #         print ("%d , %d len: %d"%(kind_val[start],kind_val[end-1],len(dftmp)))
        dfa.append(dftmp)
    ed = dt.now()
    print("inner_optimized_split_method min_num %d cosuming : %f" % (min_num, (ed - st).total_seconds()))
    # if end == len(kind_val):
    #     break
    return dfa, group_kind_vals_list, contain_null


# def split_df(df,feature_column_name,bins=10,as_type=None):
#     """
#     用来根据 feature_column_name 列值排序 把原DataFrame 分为 bins 个 或 bins+1 个（如果 列值中有空值）
#     排序为 None 第一 其余按照从小到大排列
#     df:DataFrame 传入 用于分隔
#     feature_column_name:分值列名
#     bins:划分个数 （default : 10)
#     as_type:列中非空值排序前强制转化类型 （default：float ,exm:int,float,str(not excepted)...)
#     return:
#     返回 list() feature_column_name DataFrame 子集
#     """
#
#     if as_type!=None:
#         df.feature_column_name = df.feature_column_name.astype(as_type)
#         # df.loc[not_null,feature_column_name]=df[not_null][feature_column_name].astype(as_type)
#     # print (feature_column_name)
#     not_null=~df[feature_column_name].isnull()
#     df_n=df[~not_null]#值空集合
#     df_nn=df[not_null]#值非空集合
#     reorder_df(df_nn,feature_column_name)
#     kind_val=list(df_nn.groupby(feature_column_name).size().index.values)
#     kind_size=df_nn.groupby(feature_column_name).size().values
#
#     min_num=math.ceil(len(df_nn)/bins)#每组中最小数量,向上取整
# #     print ("%s min length %d"%(feature_column_name,min_num))#打印信息用于调试
#     dfa = []
#     group_kind_vals_list=[]
#     contain_null = False
#     if len(df_n)!=0:
#         dfa.append(df_n)
#         contain_null = True
#         group_kind_vals_list.append([df_n[feature_column_name].values[0]])
#     end = 0
#     for i in range(bins):
#         start=end
#         end=binary_search_sum(kind_size,start,start,len(kind_size),min_num)
#         group_kind_vals=kind_val[start:end]
#         group_kind_vals_list.append(group_kind_vals)
# #         print (type(group_kind_vals[0]))
# #         print (type(df_nn[feature_column_name].values[0]))
#         dftmp=df_nn[df_nn[feature_column_name].isin(group_kind_vals,)]
# #         print ("kind_size length:"+str(len(kind_size))+"kind_val length: "+str(len(kind_val))+" end:"+str(end))
# #         print ("%d , %d len: %d"%(kind_val[start],kind_val[end-1],len(dftmp)))
#         dfa.append(dftmp)
#         if end == len(kind_val):
#             break
#     return dfa,group_kind_vals_list,contain_null

def get_kinds(df, column):
    return df[~df[column].duplicated()][column].values


def iv(df_i, yp_all, yn_all, y_column, positive_label_value=1):
    group_yi = (df_i[y_column] == positive_label_value).sum()
    group_yin = (df_i[y_column] != positive_label_value).sum()
    if group_yi == 0 or group_yin == 0:
        print("Warning no  postive or negtive value in this group ")
        return 0, 0
    pyi = group_yi / yp_all
    pni = group_yin / yn_all
    woe = np.log(pyi / pni)
    return (pyi - pni) * woe, woe


def range_str(df_i, column_name):
    min_val = df_i[column_name].min()
    max_val = df_i[column_name].max()
    if min_val is None:
        return None
    if "float" in str(type(min_val)):
        if min_val == max_val:
            return "%.3f" % (min_val)
        else:
            return "[ %.3f , %.3f ]" % (min_val, max_val)
    else:
        if min_val == max_val:
            return str(min_val)
        else:
            return "[ %s , %s ]" % (str(min_val), str(max_val))


class Basic(object):
    def __init__(self, df, feature_column_names, label_column_names, as_type=None, output_dir=None, pos_label=1,
                 bad_label=1, name_split_in_dic="*"):

        self.df = df
        self.feature_column_names = feature_column_names
        # self.label_like_feature_column_name=label_like_feature_column_name
        self.label_column_names = label_column_names
        self.as_type = as_type
        self.output_dir = output_dir
        self.name_split = name_split_in_dic
        self.pos_label = pos_label
        self.bad_label = bad_label
        self.pre_deal_with_DataFrame(self.df, as_type)  # important do not change this line of code
        self.basic_info = None
        # self.basic_info = self.basic_infos_by_args(pos_label=pos_label)

    def basic_infos_by_args(self, pos_label=1):
        basic_info = {}
        for column in self.feature_column_names:
            for label in self.label_column_names:
                st = dt.now()
                above_threshold_is_pos, thresholds, tpr, fpr, accuracy, recall, precision = basic_model_info(
                    self.df[column].values, self.df[label].values, pos_label=pos_label)
                dic_tmp = {}
                dic_tmp['above_threshold_is_pos'] = above_threshold_is_pos
                dic_tmp['thresholds'] = thresholds
                dic_tmp['tpr'] = tpr
                dic_tmp['fpr'] = fpr
                dic_tmp['accuracy'] = accuracy
                dic_tmp['recall'] = recall
                dic_tmp['precision'] = precision
                dic_tmp['auc'] = auc_values(fpr, tpr)
                dic_tmp['max_ks'] = (tpr - fpr).max()
                basic_info[label + self.name_split + column] = dic_tmp
                ed = dt.now()
                print("basic_infos_by_args getting basic info %s cosuming : %f seconds" % (
                column + self.name_split + label, (ed - st).total_seconds()))
        return basic_info

    def plot_all(self, max_item=-1, bins_for_dis=50):
        """
        根据初始化数据
        输出 概率密度，auc ，ks 曲线图
        """
        # 概率密度plot
        df_count_dic, df_dic = self.column_count_bins(self.feature_column_names, bins=bins_for_dis,
                                                      as_type=self.as_type)

        if self.basic_info is None:
            self.basic_infos_by_args(pos_label=pos_label)
        lf = len(self.feature_column_names)
        ll = len(self.label_column_names)
        if lf > 1:
            self.plot_auc_by_name(origin_n=self.label_column_names)
            self.plot_ks_by_name(origin_n=self.label_column_names)
            self.normal_plot("accuracy", origin_n=self.label_column_names)
            self.normal_plot("recall", origin_n=self.label_column_names)
            self.normal_plot("precision", origin_n=self.label_column_names)
        elif lf == 1:
            self.plot_auc_by_name(origin_n=self.feature_column_names)
            self.plot_ks_by_name(origin_n=self.feature_column_names)
            self.normal_plot("accuracy", origin_n=self.feature_column_names)
            self.normal_plot("recall", origin_n=self.feature_column_names)
            self.normal_plot("precision", origin_n=self.feature_column_names)

        else:
            print("do not know how to plot pics len(feature) : %d len(labels): %d" % (lf, ll))

    def normal_plot(self, plot_name, max_item=-1, origin_n=None):
        """
        一个图中一个 label 多个feature column 用于查看 在同样label 下 不同column 的 plot_name 图表
        """
        if origin_n != None:
            origin_names = origin_n
        else:
            origin_names = self.label_column_names
        for ori_name in origin_names:
            this_dic = {}
            for key, dic_t in self.basic_info.items():
                if ori_name in key:
                    if ori_name == key.split(self.name_split)[1]:
                        other_name = key.split(self.name_split)[0]
                        this_dic[other_name] = dic_t
                    else:
                        other_name = key.split(self.name_split)[1]
                        this_dic[other_name] = dic_t
            start = 0
            max_l = 0
            other_names = list(this_dic.keys())
            if max_item == -1:
                max_l = len(this_dic)
            else:
                max_l = max_item
            while start < len(other_names):

                end = start + max_l
                if end > len(other_names):
                    end = len(other_names)

                plt.figure(figsize=(15, 10))
                labels = []
                # labels.append('random,auc=0.5' )
                # plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

                for i in range(start, end):
                    this_other_name = other_names[i]
                    plt.plot(np.arange(len(this_dic[this_other_name]['thresholds'])) / len(
                        this_dic[this_other_name]['thresholds']), this_dic[this_other_name][plot_name])
                    #                 plt.text(mxindex[0]+30,mxks+rv+0.03,'max ks %s %.3f' % (lc,mxks),ha='center')
                    #                 plt.arrow(mxindex[0],mxks,30,rv)
                    labels.append("(" + str(
                        len(this_dic[this_other_name]['thresholds'])) + ') ' + ori_name + ' ' + this_other_name)

                start = start + max_l
                # plt.xlim([0.0,1.0])
                # plt.ylim([0.0,1.0])
                plt.title(plot_name + ' curve ' + ori_name)
                plt.xlabel('predict positive rate')
                plt.ylabel(plot_name)
                plt.grid(True)
                plt.legend(labels)
                if self.output_dir is not None:
                    plt.savefig(self.output_dir + ori_name + "_" + str(start) + "_" + plot_name + ".jpg")
                plt.show()

    def plot_ks_by_name(self, max_item=-1, origin_n=None):
        """
        一个图中一个 label 多个feature column 用于查看 在同样label 下 不同column 的 auc 图表
        """
        if origin_n != None:
            origin_names = origin_n
        else:
            origin_names = self.label_column_names
        for ori_name in origin_names:
            this_dic = {}
            for key, dic_t in self.basic_info.items():
                if ori_name in key:
                    if ori_name == key.split(self.name_split)[1]:
                        other_name = key.split(self.name_split)[0]
                        this_dic[other_name] = dic_t
                    else:
                        other_name = key.split(self.name_split)[1]
                        this_dic[other_name] = dic_t
            start = 0
            max_l = 0
            other_names = list(this_dic.keys())
            if max_item == -1:
                max_l = len(this_dic)
            else:
                max_l = max_item
            while start < len(other_names):

                end = start + max_l
                if end > len(other_names):
                    end = len(other_names)

                plt.figure(figsize=(15, 10))
                labels = []
                # labels.append('random,auc=0.5' )
                # plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

                for i in range(start, end):
                    this_other_name = other_names[i]
                    plt.plot(np.arange(len(this_dic[this_other_name]['thresholds'])) / len(
                        this_dic[this_other_name]['thresholds']),
                             np.abs(this_dic[this_other_name]['tpr'] - this_dic[this_other_name]['fpr']))
                    mxks = (this_dic[this_other_name]['tpr'] - this_dic[this_other_name]['fpr']).max()
                    mxindex = np.where((this_dic[this_other_name]['tpr'] - this_dic[this_other_name]['fpr']) == mxks)
                    rv = random.random() / 130
                    #                 plt.text(mxindex[0]+30,mxks+rv+0.03,'max ks %s %.3f' % (lc,mxks),ha='center')
                    #                 plt.arrow(mxindex[0],mxks,30,rv)
                    plt.text(mxindex[0] / len(this_dic[this_other_name]['tpr']), mxks * 1.015 + rv,
                             'max ks %s %.3f' % (this_other_name, mxks), ha='center')
                    plt.arrow(mxindex[0] / len(this_dic[this_other_name]['tpr']), mxks, 0, 0)
                    labels.append("(" + str(len(this_dic[this_other_name][
                                                    'thresholds'])) + ') ' + ori_name + ' ' + this_other_name + ' mxks=' + str(
                        np.round(mxks, 4)))

                start = start + max_l
                # plt.xlim([0.0,1.0])
                # plt.ylim([0.0,1.0])
                plt.title('ks curve ' + ori_name)
                plt.xlabel('predict positive rate')
                plt.ylabel('ks')
                plt.grid(True)
                plt.legend(labels)
                if self.output_dir is not None:
                    plt.savefig(self.output_dir + ori_name + "_" + str(start) + "_ks.jpg")
                plt.show()

    def plot_auc_by_name(self, max_item=-1, origin_n=None):
        """
        一个图中一个 label 多个feature column 用于查看 在同样label 下 不同column 的 auc 图表
        """
        if origin_n != None:
            origin_names = origin_n
        else:
            origin_names = self.label_column_names
        for ori_name in origin_names:
            this_dic = {}
            for key, dic_t in self.basic_info.items():
                if ori_name in key:
                    if ori_name == key.split(self.name_split)[1]:
                        other_name = key.split(self.name_split)[0]
                        this_dic[other_name] = dic_t
                    else:
                        other_name = key.split(self.name_split)[1]
                        this_dic[other_name] = dic_t
            start = 0
            max_l = 0
            other_names = list(this_dic.keys())
            # print (other_names)
            if max_item == -1:
                max_l = len(this_dic)
            else:
                max_l = max_item
            while start < len(other_names):

                end = start + max_l
                if end > len(other_names):
                    end = len(other_names)

                plt.figure(figsize=(15, 10))
                labels = []
                labels.append('random,auc=0.5')
                plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

                for i in range(start, end):
                    this_other_name = other_names[i]
                    labels.append("(" + str(len(this_dic[this_other_name][
                                                    'thresholds'])) + ') ' + ori_name + ' ' + this_other_name + ' auc=' + str(
                        np.round(this_dic[this_other_name]['auc'], 4)))
                    plt.plot(this_dic[this_other_name]['fpr'], this_dic[this_other_name]['tpr'])
                start = start + max_l
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.title('ROC curve ' + ori_name)
                plt.xlabel('False Positive Rate ( 1- specificity )')
                plt.ylabel('True Positive Rate ( Sensitivity )')
                plt.grid(True)
                plt.legend(labels)
                if self.output_dir is not None:
                    plt.savefig(self.output_dir + ori_name + "_" + str(start) + "_auc.jpg")
                plt.show()

    def pre_deal_with_DataFrame(self, df, as_type=None):
        """
        用来强制转化 df 的相关column （不建议要直接调用）
        ！！！ 传入的 df 的列属性会被改变  ！！!
        """
        if as_type != None:
            for column in as_type:
                if column in df.columns:
                    df[column] = df[column].astype(as_type[column])

    def inner_split(self, bins=10, positive_label_value=1):
        """
        iv_info 调用 内部分隔方法 不建议直接调用
        """
        split_df_dic = {}
        split_group_dic = {}
        for label in self.label_column_names:
            df_label_not_none = self.df[~self.df[label].isnull()]
            y_kinds = get_kinds(df_label_not_none, label)
            if positive_label_value not in y_kinds:
                raise Exception("positive_label_value %s not in datafrane column %s" % (positive_label_value, label))
            for column in self.feature_column_names:
                df_split_tmp, group_kind_vals_list, contain_null = optimized_split_df(df_label_not_none, column,
                                                                                      bins=bins)
                split_df_dic[label + self.name_split + column] = df_split_tmp
                split_group_dic[label + self.name_split + column] = group_kind_vals_list
        return split_df_dic, split_group_dic

    def auc_sumary(self, positive_label_value=1):
        if self.basic_info is None:
            self.basic_info = self.basic_infos_by_args(pos_label=positive_label_value)
        column_names = ['feature_name']
        column_names.extend(self.label_column_names)
        list_auc = []
        for column in self.feature_column_names:
            list_row = [column]
            for label in self.label_column_names:
                list_row.append(self.basic_info[label + self.name_split + column]['auc'])
            list_auc.append(list_row)

        df_auc = pd.DataFrame(list_auc, columns=column_names)
        return df_auc

    def iv_sumary(self, bins=10, positive_label_value=1, as_type=None):
        """
        return :
        df_iv: 盛放各个 feature 和 target label 之间的 iv 值 （DataFrame)
        """

        iv_dic, split_df_dic, split_group_dic = self.iv_info(bins=bins, positive_label_value=positive_label_value,
                                                             as_type=as_type)
        column_names = ['feature_name']
        column_names.extend(self.label_column_names)
        list_iv = []
        for column in self.feature_column_names:
            list_iv_row = [column]
            for label in self.label_column_names:
                list_iv_row.append(iv_dic[label + self.name_split + column]['iv_accumlate'].values[-1])
            list_iv.append(list_iv_row)

        df_iv = pd.DataFrame(list_iv, columns=column_names)
        return df_iv

    def iv_info(self, bins=10, positive_label_value=1, as_type=None):
        """
        return :
        iv_dic,split_df_dic,split_group_dic
        iv_dic:根据不同的 label 及 feature 计算相关 feature 的分段 IV 值 DataFrame （dic）
        split_df_dic：根据不同的 label 及 feature 获得的子分组 list （dic）
        split_group_dic：根据不同的 label 及 feature 获得的子分组值 list （dic）
        """
        self.pre_deal_with_DataFrame(self.df, as_type)
        split_df_dic, split_group_dic = self.inner_split(bins=bins, positive_label_value=positive_label_value)
        iv_dic = {}

        for k, df_l in split_df_dic.items():
            label = k.split(self.name_split)[0]
            column = k.split(self.name_split)[1]
            range_tmp = []
            count_tmp = []
            count_percent_tmp = []
            woe_tmp = []
            iv_tmp = []
            iv_accumlate_tmp = []
            yp_all = 0
            yn_all = 0
            alen = 0
            for df_i in df_l:
                alen = alen + len(df_i)
                yp_all = yp_all + (df_i[label] == positive_label_value).sum()
                yn_all = yn_all + (df_i[label] != positive_label_value).sum()
            for df_i in df_l:
                range_tmp.append(range_str(df_i, column))
                count_tmp.append(len(df_i))
                count_percent_tmp.append(len(df_i) / alen)
                iv_val, woe = iv(df_i, yp_all, yn_all, label, positive_label_value=positive_label_value)
                woe_tmp.append(woe)
                iv_tmp.append(iv_val)
                iv_accumlate_tmp.append(np.array(iv_tmp).sum())
            # dt_tmp = pd.DataFrame(np.concatenate((np.array([range_tmp],dtype=np.str).T,
            #                                              np.array([count_tmp],dtype=np.int64).T,
            #                                               np.array([count_percent_tmp],dtype=np.float64).T,
            #                                               np.array([woe_tmp],dtype=np.float64).T,
            #                                               np.array([iv_tmp],dtype=np.float64).T,
            #                                               np.array([iv_accumlate_tmp],dtype=np.float64).T,
            #                                              ),axis=1),columns=['范围','数量','百分比','woe','iv_i','iv_accumlate'])
            column_order = ['范围', '数量', '百分比', 'woe', 'iv_i', 'iv_accumlate']
            dt_tmp = pd.DataFrame({'范围': np.array(range_tmp, dtype=np.str),
                                   '数量': np.array(count_tmp, dtype=np.int64),
                                   '百分比': np.array(count_percent_tmp, dtype=np.float64),
                                   'woe': np.array(woe_tmp, dtype=np.float64),
                                   'iv_i': np.array(iv_tmp, dtype=np.float64),
                                   'iv_accumlate': np.array(iv_accumlate_tmp, dtype=np.float64)
                                   })

            iv_dic[k] = dt_tmp[column_order]
        return iv_dic, split_df_dic, split_group_dic

    def split_by_special_label_kind_val(self, df, column_names, special_label, bins=10, positive_label_value=1,
                                        as_type=None):
        """
        根据 指定label 作为基准 对所有column_names （包含None）分组,并返回 子分组DataFrame 及 相关df中含有的对应column 中的值个体（都存储在dic中）
        df:DataFrame 传入 用于分隔
        column_names:所有需要划分的feature column name
        special_label:用于指定特殊分组的基指标，基指标确定所有feature_colum 的分组值(即特殊label的值不为None)
        bins:分组个数 （default 10)
        positive_label_value:正类在 label 中的值 （只考虑2分类问题，多分类问题简化为二分类）
        as_type:column 中的值分段中将被转化为的类型，然后用于比较排序后分段
        return：
        split_df_dic,split_group_dic
        split_df_dic：字典类型 其中是list 存储某字典下子分组(子元素是 DataFrame)
        split_group_dic:字典类型 其中是list 存储某字典下子分组中的column 值
        """
        self.pre_deal_with_DataFrame(df, as_type)
        split_df_dic = {}
        split_group_dic = {}
        df_label_not_none = df[~df[special_label].isnull()]
        y_kinds = get_kinds(df_label_not_none, special_label)
        if positive_label_value not in y_kinds:
            raise Exception("positive_label_value %s not in datafrane column %s" % (positive_label_value, label))
        for column in column_names:
            if as_type != None:
                df_split_tmp, group_kind_vals_list, contain_null = optimized_split_df(df_label_not_none, column,
                                                                                      bins=bins,
                                                                                      as_type=as_type[column])
            else:
                df_split_tmp, group_kind_vals_list, contain_null = optimized_split_df(df_label_not_none, column,
                                                                                      bins=bins)
            split_df_dic[special_label + self.name_split + column] = df_split_tmp
            split_group_dic[column] = group_kind_vals_list  # 只关心分组的区间（后续分组区间与此分组一致）
        return split_df_dic, split_group_dic

    def split_by_group_column_kind_val(self, df, column_names, labels, split_group_dic, as_type=None):
        """
        根据 已经确认label 列 分组的各 column 分组值，在其他label 下 同分组分下分组。
        return：
        split_df_dic：字典类型 其中是list 存储某字典下子分组(子元素是 DataFrame)
        """
        self.pre_deal_with_DataFrame(df, as_type)
        split_df_dic = {}
        for column in column_names:
            not_null = ~df[column].isnull()
            # df.loc[not_null,column]=df[not_null][column].astype(as_type)
            group_vals_list = split_group_dic[column]
            for label in labels:
                df_label_not_none = df[(~df[label].isnull())]
                all_leng_tmp = len(df_label_not_none)
                tmp_this_out_leng = 0
                df_split_tmp = []
                for group_vals_tmp in group_vals_list:
                    df_label_not_none_group_tmp = df_label_not_none[df_label_not_none[column].isin(group_vals_tmp, )]
                    tmp_this_out_leng = tmp_this_out_leng + len(df_label_not_none_group_tmp)
                    df_split_tmp.append(df_label_not_none_group_tmp)
                # addition_val = False
                if tmp_this_out_leng != all_leng_tmp:
                    #                     addition_val = True
                    print("Warning :label :%s  column :%s has additional group val leng not match diff count %d" % (
                    label, column, all_leng_tmp - tmp_this_out_leng))
                split_df_dic[label + self.name_split + column] = df_split_tmp
        return split_df_dic

    def split_by_special_label_and_columns(self, check_column_names, label_columns, special_label, bins=10,
                                           positive_label_value=1):
        """
        根据 special_label 作为基准 check_column_names （包含None）分组,并返回 子分组DataFrame 及 相关df中含有的对应 check_column_names 中的值个体（都存储在dic中）
        df:DataFrame 传入 用于分隔
        check_column_names:所有需要划分的feature column name
        label_columns:所有非基准label 根据 special_label 的分组分相应 check_column_names 的分组值
        special_label:用于指定特殊分组的基指标，基指标确定所有 check_column_names 的分组值(即特殊label的值不为None)
        bins:分组个数 （default 10)
        positive_label_value:正类在 label 中的值 （只考虑2分类问题，多分类问题简化为二分类）
        as_type:column 中的值分段中将被转化为的类型，然后用于比较排序后分段
        return：
        """
        split_df_dic, split_group_dic = self.split_by_special_label_kind_val(self.df, check_column_names, special_label,
                                                                             bins=bins,
                                                                             positive_label_value=positive_label_value)
        if special_label in label_columns:
            label_columns.remove(special_label)
        split_df_dic_left = self.split_by_group_column_kind_val(self.df, check_column_names, label_columns,
                                                                split_group_dic)

        split_df_dic_merge = split_df_dic.update(split_df_dic_left)
        return split_df_dic, split_group_dic

    def get_special_df(self, split_df_dic, name):
        for key, val in split_df_dic.items():
            if name in key:
                return val
        raise Exception("%s not in split_df_dic keys : %s " % (name, split_df_dic.keys()))

    def column_count_bins(self, check_column_names, bins=50, as_type=None, plot=True):
        """
        等宽分组 获取分组数量
        """
        self.pre_deal_with_DataFrame(self.df, as_type)
        df_dic = {}
        df_count_dic = pd.DataFrame()
        for column in check_column_names:
            df_list = []
            x_tmp = []
            y_tmp = []
            y_percent_tmp = []
            type_c = str(self.df[column].dtype)
            if ("int" not in type_c) and ("float" not in type_c):
                raise Exception("not supported dtype %s in column : %s" % (type_c, column))
            else:
                min_v = self.df[column].min()
                max_v = self.df[column].max()
                space = (max_v - min_v) / bins
                start = min_v

                for i in range(bins):
                    if "float" not in type_c:
                        end_this = math.ceil(start + space)
                    else:
                        end_this = start + space
                    if i != (bins - 1):
                        x_tmp.append("[" + str(start) + "," + str(end_this) + ")")
                        df_i = self.df[(self.df[column] >= start) & (self.df[column] < (end_this))]
                    else:
                        x_tmp.append("[" + str(start) + "," + str(max_v) + "]")
                        df_i = self.df[(self.df[column] >= start) & (self.df[column] <= (max_v))]
                    start = end_this
                    df_list.append(df_i)
                    y_tmp.append(len(df_i))
                    y_percent_tmp.append(len(df_i) / len(self.df))
            df_count_dic[column + ' range'] = np.array(x_tmp, dtype=np.str)
            df_count_dic[column + ' count'] = np.array(y_tmp, dtype=np.int64)
            df_count_dic[column + ' percentage'] = np.array(y_percent_tmp, dtype=np.float64)
            df_dic[column] = df_list
            if plot:
                plot_data(x_tmp, y_tmp, column)
                plot_data(x_tmp, y_percent_tmp, column)
        return df_count_dic, df_dic

    def column_labels_distribution2(self, check_column_names, kind_colum_name, order=None, bins=10, plot=True):
        """
        optimized 分类百分比比例 分组 并画图 （推荐）
        bins = -1 每类画图
        """
        dic_label_column_df = {}
        for check_column_name in check_column_names:
            dfa, group_kind_vals_list, contain_null = optimized_split_df(self.df, check_column_name, bins=bins)

            kind_val = list(self.df.groupby(kind_colum_name).size().index.values)
            if order is not None:
                for oi in order:
                    if oi not in kind_val:
                        raise Exception("order vale %s not in kind_val" % oi)
            if contain_null:
                if None not in order:
                    order.append(None)
            range_tmp = []  # 列值
            show_y = []

            for df_i in dfa:
                range_tmp.append(range_str(df_i, check_column_name))
            df_tmp = pd.DataFrame({check_column_name + " range": np.array(range_tmp, dtype=np.str)})
            for kind in order:
                count_tmp = []
                count_percent_tmp = []
                label_distri_tmp = []
                all_length = 0
                for df_i in dfa:
                    len_of_temp = (df_i[kind_colum_name] == kind).sum()
                    all_length = all_length + len_of_temp
                    if len_of_temp == 0:
                        count_tmp.append(0)
                        count_percent_tmp.append(0)
                        label_distri_tmp.append(None)
                    else:
                        count_tmp.append(len_of_temp)
                        count_percent_tmp.append(len_of_temp / all_length)
                        label_distri_tmp.append(len_of_temp / len(df_i))

                df_tmp[kind + "数量"] = np.array(count_tmp, dtype=np.int64)
                df_tmp[kind + "百分比"] = np.array(count_percent_tmp, dtype=np.float64)
                df_tmp[kind + " inner percent"] = np.array(label_distri_tmp, dtype=np.float64)
                show_y.append(kind + " inner percent")
            reorder_columns_names = [check_column_name + " range"]
            for cn in df_tmp.columns:
                if "数量" in cn:
                    reorder_columns_names.append(cn)
            for cn in df_tmp.columns:
                if "百分比" in cn:
                    reorder_columns_names.append(cn)
            for cn in df_tmp.columns:
                if ("数量" not in cn) and ("百分比" not in cn) and ("range" not in cn):
                    reorder_columns_names.append(cn)
            dic_label_column_df[check_column_name] = df_tmp[reorder_columns_names]

            if plot:
                self.plot_data_normal(dic_label_column_df, show_y)
            for key, df_key in dic_label_column_df.items():
                for cn in df_key.columns:
                    if 'float' in str(df_key[cn].dtype):
                        df_key[cn] = df_key[cn].apply(percent_str)
        return dic_label_column_df

    def column_labels_distribution(self, check_column_names, label_columns, special_label, bins=10,
                                   positive_label_value=1, as_type=None, plot=True):
        """ wait for optimization（不推荐）
            （推荐使用column_labels_distribution2）
        """
        self.pre_deal_with_DataFrame(self.df, as_type)

        ori_columns = []
        lc_tmp = []
        for i in label_columns:
            if special_label != i:
                lc_tmp.append(i)
            ori_columns.append(i)
        label_columns = lc_tmp

        split_df_dic, split_group_dic = self.split_by_special_label_and_columns(check_column_names, label_columns,
                                                                                special_label, bins=bins,
                                                                                positive_label_value=positive_label_value)
        dic_label_column_df = {}
        for column in split_group_dic:
            # column_names=[column+" range"]#列名
            # column_label_num=[]#列名
            # column_label_percent=[]#列名
            # column_label_mean=[]#列名

            range_column = []  # 列值
            count_column = []  # 列值
            count_percent_column = []  # 列值
            label_mean_column = []  # 列值

            label_tmp = special_label
            special_df_list = split_df_dic[label_tmp + self.name_split + column]
            # print (special_df_list)

            # column_label_num.append(label_tmp+"数量")
            # column_label_percent.append(label_tmp+"百分比")
            # column_label_mean.append(label_tmp)
            range_tmp = []
            count_tmp = []
            count_percent_tmp = []
            label_mean_tmp = []
            all_length = 0
            for df_i_special in special_df_list:
                len_of_temp = (df_i_special[label_tmp] == positive_label_value).sum()
                all_length = all_length + len_of_temp

            for df_i_special in special_df_list:
                len_of_temp = (df_i_special[label_tmp] == positive_label_value).sum()
                if len_of_temp == 0:
                    range_tmp.append(range_str(df_i_special, column))
                    count_tmp.append(0)
                    count_percent_tmp.append(0)
                    label_mean_tmp.append(None)
                else:
                    range_tmp.append(range_str(df_i_special, column))
                    count_tmp.append(len_of_temp)
                    count_percent_tmp.append(len_of_temp / all_length)
                    label_mean_tmp.append(df_i_special[label_tmp].mean())
            df_tmp = pd.DataFrame({column + " range": np.array(range_tmp, dtype=np.str)})
            df_tmp[label_tmp + "数量"] = np.array(count_tmp, dtype=np.int64)
            df_tmp[label_tmp + "百分比"] = np.array(count_percent_tmp, dtype=np.float64)
            # df_tmp[label_tmp+"百分比"]=df_tmp[label_tmp+"百分比"].apply(percent_str)
            df_tmp[label_tmp] = np.array(label_mean_tmp, dtype=np.float64)
            # df_tmp[label_tmp]=df_tmp[label_tmp].apply(percent_str)
            # range_column.append(range_tmp)
            # count_column.append(count_tmp)
            # count_percent_column.append(count_percent_tmp)
            # label_mean_column.append(label_mean_tmp)
            for label in label_columns:
                df_list_i = split_df_dic[label + self.name_split + column]
                label_tmp = label
                # column_label_num.append(label_tmp+"数量")
                # column_label_percent.append(label_tmp+"百分比")
                # column_label_mean.append(label_tmp)
                count_tmp = []
                count_percent_tmp = []
                label_mean_tmp = []
                all_length = 0
                for df_i in df_list_i:
                    len_of_temp = (df_i[label_tmp] == positive_label_value).sum()
                    all_length = all_length + len_of_temp
                for df_i in df_list_i:
                    len_of_temp = (df_i[label_tmp] == positive_label_value).sum()
                    count_tmp.append(len_of_temp)
                    count_percent_tmp.append(len_of_temp / all_length)
                    label_mean_tmp.append(df_i[label_tmp].mean())
                df_tmp[label_tmp + "数量"] = np.array(count_tmp, dtype=np.int64)
                df_tmp[label_tmp + "百分比"] = np.array(count_percent_tmp, dtype=np.float64)
                # df_tmp[label_tmp+"百分比"]=df_tmp[label_tmp+"百分比"].apply(percent_str)
                df_tmp[label_tmp] = np.array(label_mean_tmp, dtype=np.float64)
                # df_tmp[label_tmp]=df_tmp[label_tmp].apply(percent_str)
                # count_column.append(count_tmp)
                # count_percent_column.append(count_percent_tmp)
                # label_mean_column.append(label_mean_tmp)
            # column_names.extend(column_label_num)
            # column_names.extend(column_label_percent)
            # column_names.extend(column_label_mean)
            # dt_tmp = pd.DataFrame(np.concatenate((np.array(range_column,dtype=np.str).T,
            #                                              np.array(count_column,dtype=np.int64).T,
            #                                               np.array(count_percent_column,dtype=np.float64).T,
            #                                               np.array(label_mean_column,dtype=np.float64).T,
            #                                              ),axis=1),columns=column_names)
            reorder_columns_names = [column + " range"]
            for cn in df_tmp.columns:
                if "数量" in cn:
                    reorder_columns_names.append(cn)
            for cn in df_tmp.columns:
                if "百分比" in cn:
                    reorder_columns_names.append(cn)
            for cn in df_tmp.columns:
                if ("数量" not in cn) and ("百分比" not in cn) and ("range" not in cn):
                    reorder_columns_names.append(cn)
            dic_label_column_df[column] = df_tmp[reorder_columns_names]
        if plot:
            self.plot_data_normal(dic_label_column_df, ori_columns)
        for key, df_key in dic_label_column_df.items():
            for cn in df_key.columns:
                if 'float' in str(df_key[cn].dtype):
                    df_key[cn] = df_key[cn].apply(percent_str)
        return dic_label_column_df

    def column_labels_mean(self, check_column_names, label_columns, special_label, bins=10, positive_label_value=1,
                           as_type=None, plot=True, plot_type_mean=True):
        """
        根据 special_label 作为基准 check_column_names （包含None）分组,并返回 子分组DataFrame 及 相关df中含有的对应 check_column_names 中的值个体（都存储在dic中）

        return :
        dic_label_column_df
        根据column 分组 获得不同target 的相关参数及计算值
        """

        self.pre_deal_with_DataFrame(self.df, as_type)

        ori_columns = []
        lc_tmp = []
        for i in label_columns:
            if special_label != i:
                lc_tmp.append(i)
            ori_columns.append(i)
        label_columns = lc_tmp

        split_df_dic, split_group_dic = self.split_by_special_label_and_columns(check_column_names, label_columns,
                                                                                special_label, bins=bins,
                                                                                positive_label_value=positive_label_value)
        dic_label_column_df = {}
        for column in split_group_dic:
            # column_names=[column+" range"]#列名
            # column_label_num=[]#列名
            # column_label_percent=[]#列名
            # column_label_mean=[]#列名

            range_column = []  # 列值
            count_column = []  # 列值
            count_percent_column = []  # 列值
            label_mean_column = []  # 列值

            label_tmp = special_label
            special_df_list = split_df_dic[label_tmp + self.name_split + column]
            # print (special_df_list)

            # column_label_num.append(label_tmp+"数量")
            # column_label_percent.append(label_tmp+"百分比")
            # column_label_mean.append(label_tmp)
            range_tmp = []
            count_tmp = []
            count_percent_tmp = []
            label_mean_tmp = []
            all_length = 0
            for df_i_special in special_df_list:
                all_length = all_length + len(df_i_special)

            for df_i_special in special_df_list:
                if len(df_i_special) == 0:
                    range_tmp.append(range_str(df_i_special, column))
                    count_tmp.append(0)
                    count_percent_tmp.append(0)
                    label_mean_tmp.append(None)
                else:
                    range_tmp.append(range_str(df_i_special, column))
                    count_tmp.append(len(df_i_special))
                    count_percent_tmp.append(len(df_i_special) / all_length)
                    label_mean_tmp.append(df_i_special[label_tmp].mean())
            df_tmp = pd.DataFrame({column + " range": np.array(range_tmp, dtype=np.str)})
            df_tmp[label_tmp + "数量"] = np.array(count_tmp, dtype=np.int64)
            df_tmp[label_tmp + "百分比"] = np.array(count_percent_tmp, dtype=np.float64)
            # df_tmp[label_tmp+"百分比"]=df_tmp[label_tmp+"百分比"].apply(percent_str)
            df_tmp[label_tmp] = np.array(label_mean_tmp, dtype=np.float64)
            # df_tmp[label_tmp]=df_tmp[label_tmp].apply(percent_str)
            # range_column.append(range_tmp)
            # count_column.append(count_tmp)
            # count_percent_column.append(count_percent_tmp)
            # label_mean_column.append(label_mean_tmp)
            for label in label_columns:
                print(split_df_dic.keys())
                df_list_i = split_df_dic[label + self.name_split + column]
                label_tmp = label
                # column_label_num.append(label_tmp+"数量")
                # column_label_percent.append(label_tmp+"百分比")
                # column_label_mean.append(label_tmp)
                count_tmp = []
                count_percent_tmp = []
                label_mean_tmp = []
                all_length = 0
                for df_i in df_list_i:
                    all_length = all_length + len(df_i)
                for df_i in df_list_i:
                    count_tmp.append(len(df_i))
                    count_percent_tmp.append(len(df_i) / all_length)
                    label_mean_tmp.append(df_i[label_tmp].mean())
                df_tmp[label_tmp + "数量"] = np.array(count_tmp, dtype=np.int64)
                df_tmp[label_tmp + "百分比"] = np.array(count_percent_tmp, dtype=np.float64)
                # df_tmp[label_tmp+"百分比"]=df_tmp[label_tmp+"百分比"].apply(percent_str)
                df_tmp[label_tmp] = np.array(label_mean_tmp, dtype=np.float64)
                # df_tmp[label_tmp]=df_tmp[label_tmp].apply(percent_str)
                # count_column.append(count_tmp)
                # count_percent_column.append(count_percent_tmp)
                # label_mean_column.append(label_mean_tmp)
            # column_names.extend(column_label_num)
            # column_names.extend(column_label_percent)
            # column_names.extend(column_label_mean)
            # dt_tmp = pd.DataFrame(np.concatenate((np.array(range_column,dtype=np.str).T,
            #                                              np.array(count_column,dtype=np.int64).T,
            #                                               np.array(count_percent_column,dtype=np.float64).T,
            #                                               np.array(label_mean_column,dtype=np.float64).T,
            #                                              ),axis=1),columns=column_names)
            reorder_columns_names = [column + " range"]
            for cn in df_tmp.columns:
                if "数量" in cn:
                    reorder_columns_names.append(cn)
            for cn in df_tmp.columns:
                if "百分比" in cn:
                    reorder_columns_names.append(cn)
            for cn in df_tmp.columns:
                if ("数量" not in cn) and ("百分比" not in cn) and ("range" not in cn):
                    reorder_columns_names.append(cn)
            dic_label_column_df[column] = df_tmp[reorder_columns_names]
        if plot:
            if plot_type_mean:
                self.plot_data(dic_label_column_df, ori_columns)
            else:
                self.plot_data_normal(dic_label_column_df, ori_columns)
        for key, df_key in dic_label_column_df.items():
            for cn in df_key.columns:
                if 'float' in str(df_key[cn].dtype):
                    df_key[cn] = df_key[cn].apply(percent_str)
        return dic_label_column_df

    def plot_data_normal(self, df_dic, y_columns):
        for k, df in df_dic.items():
            x_column = k
            fig = plt.figure(figsize=[15, 10])
            df = df.fillna(0)
            df = df.replace('nan', 0)
            bins = df[x_column + " range"].values
            for y in y_columns:
                y_tmp = df[y].astype(float).values
                buttom_df = df[y_columns[:y_columns.index(y)]]
                buttom_tmp_v = np.array(list(buttom_df.sum(axis=1)))
                df_opd = pd.DataFrame([y_tmp], columns=bins)
                x = np.arange(len(bins))
                plt.bar(x, y_tmp, bottom=buttom_tmp_v, label=y, tick_label=bins)
                if len(bins) > 40:
                    for a, b, c in zip(x, buttom_tmp_v + y_tmp / 2, y_tmp):
                        plt.text(a, b, '%.2f' % c, ha='center')
                elif len(bins) > 20:
                    for a, b, c in zip(x, buttom_tmp_v + y_tmp / 2, y_tmp):
                        plt.text(a, b, '%.3f' % c, ha='center')
                else:
                    for a, b, c in zip(x, buttom_tmp_v + y_tmp / 2, y_tmp):
                        plt.text(a, b, '%.4f' % c, ha='center')
            plt.legend()
            plt.title(x_column + ' and distribution')
            plt.ylabel('distribution')
            plt.xticks(rotation=70)
            if self.output_dir is not None:
                plt.savefig(self.output_dir + x_column + "_rate_at_target.jpg")
            plt.show()

    def plot_data(self, df_dic, y_columns):
        """用来 绘制 column_labels_mean 函数 得到 的 DataFrame


        """
        for k, df in df_dic.items():
            x_column = k
            fig = plt.figure(figsize=[15, 10])
            df = df.fillna(0)
            df = df.replace('nan', 0)
            bins = df[x_column + " range"].values
            for y in y_columns:
                y_tmp = df[y].astype(float).values
                df_opd = pd.DataFrame([y_tmp], columns=bins)
                x = np.arange(len(bins))
                plt.bar(x, y_tmp, alpha=0.5, label=y, tick_label=bins)
                for a, b in zip(x, y_tmp):
                    plt.text(a, b + 0.003, '%.4f' % b, ha='center')
                df_opd.iloc[0, :].plot(linestyle='dashdot', marker='o', label=y)
            plt.legend()
            plt.title(x_column + ' and delay rate at target')
            plt.ylabel('delay rate at target')
            plt.xticks(rotation=70)
            if self.output_dir is not None:
                plt.savefig(self.output_dir + x_column + "_rate_at_target.jpg")
            plt.show()
