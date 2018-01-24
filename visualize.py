import os
import sys

path = os.path.join(os.path.abspath('').rsplit('knjk', 1)[0], 'knjk')
sys.path.append(path)
import subprocess
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn import metrics as mr
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import traceback
from utils import feature_evaluation


def visualize_tree(dt, path=None, feature_names=None):
    """
    决策树可视化
    :param dt: sklearn.tree.DecisionTreeClassifier().fit后的实例
    :param path: 图片保存父路径
    :param feature_names: 特征名，默认取data.columns
    :return:
    """
    if path is None:
        dot_path = 'dt.dot'
        png_path = 'dt.png'
    else:
        dot_path = os.path.join(path, 'dt.dot')
        png_path = os.path.join(path, 'dt.png')

    if feature_names is None:
        feature_names = ['feature_%s' % i for i in range(dt.n_features_)]

    with open(dot_path, 'w') as f:
        export_graphviz(dt, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", dot_path, "-o", png_path]
    try:
        subprocess.check_call(command)
    except Exception as e:
        print(traceback.format_exc())
        print("Could not run dot, ie graphviz, to produce visualization")


def pr_curve(y_true, y_pred, ax):
    p, r, _ = mr.precision_recall_curve(y_true, y_pred)
    f1 = 2 * p * r / (p + r)

    lw = 2
    ax.plot(r, p, color='darkblue',
            lw=lw, label='P-R curve')
    ax.plot(r, f1, color='darkred',
            lw=lw, label='F1 curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall Rate')
    ax.set_ylabel('Precision Rate')
    ax.set_title('P-R curve')
    ax.legend(loc="upper left")



def roc_curve(y_true, y_pred, ax, label_type='', color='darkorange', loc=4):
    fpr, tpr, _ = mr.roc_curve(y_true, y_pred)
    auc = mr.roc_auc_score(y_true, y_pred)

    lw = 2
    ax.plot(fpr, tpr, color=color,
            lw=lw, label=label_type + ' ROC curve (auc = %0.2f)' % auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.legend(loc=loc)


def ks_curve(y_true, y_pred, ax, bins=1000):
    df = pd.DataFrame()
    df['y_true'] = np.array(y_true)
    df['y_pred'] = np.array(y_pred)
    df = df.sort_values('y_pred', ascending=False)

    if df.shape[0] > bins:
        step = int(df.shape[0] / bins)
    else:
        step = 1

    pos_num = df['y_true'].sum()
    neg_num = df.shape[0] - pos_num
    prob = []
    pos = []
    neg = []
    ks = []

    for i in range(0, df.shape[0], step):
        temp = df.iloc[:i, :]
        p_l = temp[(temp['y_true'] == 1)].shape[0]
        n_l = temp[(temp['y_true'] == 0)].shape[0]
        pos.append(p_l / pos_num)
        neg.append(n_l / neg_num)
        prob.append(i / df.shape[0])
        ks.append(pos[-1] - neg[-1])

    if i < df.shape[0]:
        i = df.shape[0]
        temp = df.iloc[:i, :]
        p_l = temp[(temp['y_true'] == 1)].shape[0]
        n_l = temp[(temp['y_true'] == 0)].shape[0]
        pos.append(p_l / pos_num)
        neg.append(n_l / neg_num)
        prob.append(i / df.shape[0])
        ks.append(pos[-1] - neg[-1])

    threshold = prob[np.argmax(np.array(pos) - np.array(neg))]
    #     max_ks = np.max(np.array(pos) - np.array(neg))
    max_ks = ks_2samp(df['y_pred'][df['y_true'] == 1], df['y_pred'][df['y_true'] == 0])[0]

    lw = 2
    ax.plot(prob, pos, color='darkorange',
            lw=lw, label='TPR')
    ax.plot(prob, neg, color='darkblue',
            lw=lw, label='FPR')
    ax.plot(prob, ks, color='darkred',
            lw=lw, label='KS (%0.2f)' % max_ks)
    ax.plot([threshold, threshold], [0, 1], color='lightgreen', lw=lw, linestyle='--',
            label='threshold (%0.2f)' % threshold)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title('K-S curve')
    ax.legend(loc="upper left")
    return threshold, max_ks


def boost_hist_curve(x, y, ax, target=None, max_depth=5, min_samples_leaf=0.01, method='avg', cut_points=None,rotation=55):
    if target is None:
        target = ''
    df = pd.DataFrame()
    if method != 'avg':
        th = feature_evaluation.cut_points(x, y, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        x2 = feature_evaluation.feature_discretion(x, th)
    else:
        if cut_points is None:
            x2 = pd.qcut(np.array(x), q=10)
        else:
            x2 = pd.cut(np.array(x), bins=cut_points, include_lowest=True)
    df['x'] = np.array(x2)
    df['y'] = np.array(y)
    t = pd.pivot_table(df, values='y', index='x', aggfunc=[np.mean, len]).reset_index().sort_values('x',
                                                                                                    ascending=False)
    t.columns = ['x', 'y', 'l']

    ax.bar(range(t.shape[0]), t.l, tick_label=t.x, color='lightblue', label='volume', yerr=0.00001, width=0.5)
    ax.set_ylim([0, max(t.l)*1.2])
    ax.set_ylabel('volume')
    # ax.legend(loc="upper right")
    ax.set_xticklabels(t.x, rotation=rotation)

    ax2 = ax.twinx()

    ax2.plot(range(t.shape[0]), t.y, color='r', label='test dataset')
    ax.set_xticklabels(t.x, rotation=rotation)
    ax2.set_title('sorting ability for %s' % target)
    ax2.set_ylabel('%s Delay Rate' % target)
    ax2.legend(loc="upper right")


def eva_plot(y_true, y_pred, bins=1000, path=None, target=None, cut_points=None, rotation=55):
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(221)
    roc_curve(y_true, y_pred, ax)
    ax = plt.subplot(222)
    ks_curve(y_true, y_pred, ax, bins=bins)
    ax = plt.subplot(223)
    pr_curve(y_true, y_pred, ax)
    ax = plt.subplot(224)
    boost_hist_curve(y_pred, y_true, ax, target=target, cut_points=cut_points, rotation=rotation)
    plt.tight_layout()

    if not os.path.exists('report'):
        os.mkdir('report')
    if path is None:
        path = 'eva_plot.png'
    plt.savefig(os.path.join('report', path))
    plt.show()


def roc_line(train, valid, test, path=None):
    figure, ax = plt.subplots()
    roc_curve(train['y_true'], train['y_pred'], color='red', label_type='train', ax=ax)
    roc_curve(valid['y_true'], valid['y_pred'], color='blue', label_type='valid', ax=ax)
    roc_curve(test['y_true'], test['y_pred'], color='yellow', label_type='test', ax=ax)
    if not os.path.exists('report'):
        os.mkdir('report')
    if path is None:
        path = 'eva_plot.png'
    plt.savefig(os.path.join('report', path))
    plt.show()

if __name__ == '__main__':
    y_t = np.random.randint(0, 2, 100)
    y_p = np.random.uniform(0, 1, 100)
    f, ax = plt.subplots(dpi = 100)
    boost_hist_curve(y_p, y_t, ax)
    plt.show()
    # roc_curve(y_t, y_p, ax)


