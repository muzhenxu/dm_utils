import pandas as pd
import numpy as np
from sklearn import feature_selection
from collections import defaultdict
import os
import math
from sklearn.metrics import classification_report

def chi2_calc(df_rule, labels, columns=None, save=True, path='datasource/chi2_result/', encoding='gbk'):
    if columns is None:
        columns = [c for c in df_rule.columns if c not in labels]

    dic_df = {}
    for t in labels:
        dic = defaultdict(dict)
        l = np.array((df_rule[t] > 0).astype(int))
        for c in columns:
            dic['命中样本数'][c] = df_rule[c].sum()
            dic['命中样本占比'][c] = df_rule[c].mean()
            dic['命中样本%s逾期率' % t][c] = l[np.where(df_rule[c] == 1)].mean()
            dic['未命中样本%s逾期率' % t][c] = l[np.where(df_rule[c] == 0)].mean()
            dic['总样本%s逾期率' % t][c] = l.mean()
            dic['p值'][c] = feature_selection.chi2(df_rule[c].reshape(-1, 1), l)[1][0]

        df_p = pd.DataFrame(dic)
        df_p = df_p[['总样本%s逾期率' % t, '命中样本%s逾期率' % t, '未命中样本%s逾期率' % t, 'p值', '命中样本占比', '命中样本数']].sort_values('p值')

        dic_df[f'rule_chi2_{t}'] = df_p

        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            for k, df_p in dic_df.items():
                # df_p.to_pickle(path + f'{k}.pkl')
                df_p.to_csv(f'{k}.csv', encoding=encoding)

    return dic_df

class Ripperk(object):
    def __init__(self, prun_ratio=0.2, dl_threshold=64, k=2, sample_threshold=100, diff_threshold=0.0):
        self.prun_ratio = prun_ratio
        self.dl_threshold = dl_threshold
        self.k = k
        self.sample_threshold = sample_threshold
        self.diff_threshold = diff_threshold

    def fit(self, df, label):
        self.rulesets = {}

        self._get_conditions(df)

        items =  list(label.value_counts().sort_values(ascending=False).index)
        self.items = list(items)

        while len(items) > 1:
            # get cls from end to start, from small to big
            item = items.pop()
            pos = df[label==item]
            neg = df[label!=item]

            ruleset = self.irep(pos, neg)

            for _ in range(self.k):
                ruleset = self.optimize(pos, neg, ruleset)

            df = self.remove_cases(df, ruleset)

            self.rulesets[item] = ruleset

    def predict(self, df):
        labels = np.array([self.items[0]] * df.shape[0])

        index_bool = np.array([True] * df.shape[0])
        for item in self.items[1:][::-1]:
            item_bool = self.bindings(df, self.rulesets[item])
            item_bool &= index_bool
            labels[item_bool] = item
            index_bool &= ~item_bool

        return labels

    def irep(self, pos, neg):
        rule_set = []
        rule = {}

        min_dl = self.init_dl

        while pos.shape[0] > 0:
            print('rule:', rule, len(pos), len(neg))

            pos_chunk = int((1 - self.prun_ratio) * pos.shape[0])
            neg_chunk = int((1 - self.prun_ratio) * neg.shape[0])

            pos_grow = pos.iloc[:pos_chunk, :]
            neg_grow = neg.iloc[:neg_chunk, :]
            rule = self.grow_rule(pos_grow, neg_grow)
            if not rule:
                return rule_set

            if self.prun_ratio > 0:
                pos_prun = pos.iloc[pos_chunk:, :]
                neg_prun = neg.iloc[neg_chunk:, :]
                rule = self.prun_rule(pos_prun, neg_prun, rule)

            rule_dl = self.dl(rule)
            if min_dl + self.dl_threshold < rule_dl:
                return rule_set
            else:
                rule_set.append(rule)
                if rule_dl < min_dl:
                    min_dl = rule_dl

                pos = self.remove_cases(pos, [rule])
                neg = self.remove_cases(neg, [rule])
        return rule_set

    def foil(self, pos, neg, condition, rule=None, ruleset=None):
        if ruleset is None:
            ruleset = []
        if rule is None:
            rule = {}
        ruleset.append(rule)

        if ruleset:
            p0 = np.sum(self.bindings(pos, ruleset))
            n0 = np.sum(self.bindings(neg, ruleset))
        else:
            p0 = len(pos)
            n0 = len(neg)

        ruleset.pop()

        new_rule = dict(rule)
        new_rule[condition[0]] = condition[1]

        ruleset.append(new_rule)

        p1 = np.sum(self.bindings(pos, ruleset))
        n1 = np.sum(self.bindings(neg, ruleset))

        ruleset.pop()

        if p1 < self.sample_threshold:
            return -10000

        if p0 == 0:
            d0 = 0
        else:
            d0 = float(p0) / (float(p0) + float(n0))

        if p1 == 0:
            d1 = 0
        else:
            d1 = float(p1) / (float(p1) + float(n1))

        return math.log(p1, 10) * (d1 - d0 - self.diff_threshold)

    def grow_rule(self, pos, neg, rule=None, ruleset=None):
        if ruleset is None:
            ruleset = []
        if rule is None:
            rule = {}

        pos = self.remove_cases(pos, ruleset)
        neg = self.remove_cases(neg, ruleset)

        while True:
            max_gain = -10000
            max_condition = None

            for condition in self.conditions:
                if condition[0] in rule:
                    continue

                gain = self.foil(pos, neg, condition, rule, ruleset)
                if max_gain < gain:
                    max_gain = gain
                    max_condition = condition

            print('condition:', max_condition, max_gain, rule)

            if max_gain <= 0:
                return rule

            rule[max_condition[0]] = max_condition[1]
            ruleset.append(rule)

            if np.sum(self.bindings(neg, ruleset)) == 0:
                ruleset.pop()
                return rule

            ruleset.pop()

    def prun_rule(self, pos, neg, rule, ruleset=None):
        if ruleset is None:
            ruleset = []

        # Deep copy our rule.
        tmp_rule = dict(rule)
        # Append the rule to the rules list.
        ruleset.append(tmp_rule)

        p = np.sum(self.bindings(pos, ruleset))
        n = np.sum(self.bindings(neg, ruleset))

        # TODO: 无效rule为何不直接返回空dict{}
        if p == 0 and n == 0:
            ruleset.pop()
            return tmp_rule

        max_rule = dict(tmp_rule)
        max_score = (p - n) / float(p + n)

        keys = list(max_rule.keys())
        i = -1

        while len(tmp_rule.keys()) > 1:
            # Remove the last attribute.
            # 这里的删减是有序的。但是grow过程的condtition学习真的可以保证先学到的比后学到的好么？
            del tmp_rule[keys[i]]

            # Recalculate score.
            p = np.sum(self.bindings(pos, ruleset))
            n = np.sum(self.bindings(neg, ruleset))

            tmp_score = (p - n) / float(p + n)

            # We found a new max score, save rule.
            if tmp_score > max_score:
                max_rule = dict(tmp_rule)
                max_score = tmp_score

            i -= 1

        # Remove the rule from the rules list.
        ruleset.pop()

        return max_rule

    def optimize(self, pos, neg, ruleset):
        new_ruleset = list(ruleset)

        pos_chunk = int((1 - self.prun_ratio) * pos.shape[0])
        neg_chunk = int((1 - self.prun_ratio) * neg.shape[0])

        pos_grow = pos.iloc[:pos_chunk, :]
        neg_grow = neg.iloc[:neg_chunk, :]

        if self.prun_ratio > 0:
            pos_prun = pos.iloc[pos_chunk:, :]
            neg_prun = neg.iloc[neg_chunk:, :]

        i = 0
        while i < len(new_ruleset):
            rule = new_ruleset.pop(i)

            reprule = self.grow_rule(pos_grow, neg_grow)
            if self.prun_ratio > 0:
                reprule = self.prun_rule(pos_prun, neg_prun, reprule, new_ruleset)

            # greedily on whole dataset
            revrule = self.grow_rule(pos, neg, rule, new_ruleset)

            rule_dl = self.dl(rule)
            reprule_dl = self.dl(reprule)
            revrule_dl = self.dl(revrule)

            if (reprule_dl < rule_dl and reprule_dl < revrule_dl):
                # Don't allow duplicates.
                if not reprule in new_ruleset:
                    new_ruleset.insert(i, reprule)
            elif (revrule_dl < rule_dl and revrule_dl < reprule_dl):
                # Don't allow duplicates.
                if not revrule in new_ruleset:
                    new_ruleset.insert(i, revrule)
            else:
                # Don't allow duplicates.
                if not rule in new_ruleset:
                    new_ruleset.insert(i, rule)

            i+= 1

        return new_ruleset

    def dl(self, rule):
        """
        Finds the description length for a rule.

        Key arguments:
        rule -- the rule.
        """
        k = len(rule.keys())
        p = k / float(self.init_dl)

        p1 = float(k) * math.log(1 / p, 2)
        p2 = float(self.init_dl - k) * math.log(1 / float(1 - p), 2)

        return int(0.5 * (math.log(k, 2) + p1 + p2))

    def _get_conditions(self, df):
        s = df.dtypes
        discrete_cols = list(s.index[s=='object'])
        category_cols = list(s.index[s=='category'])
        continuous_cols = [i for i in df.columns if i not in discrete_cols + category_cols]

        conditions = []

        for c in discrete_cols:
            for v in df[c].unique():
                conditions.append((c, ('==', v)))
                conditions.append((c, ('!=', v)))

        for c in continuous_cols:
            _, r = pd.qcut(df[c], q=100, retbins=True, duplicates='drop')
            for v in r:
                conditions.append((c, ('>=', v)))
                conditions.append((c, ('<=', v)))

        for c in category_cols:
            for v in df[c].unique():
                conditions.append((c, ('>=', v)))
                conditions.append((c, ('<=', v)))

        self.conditions = conditions
        self.init_dl = len(conditions)

    def bindings(self, df, ruleset):
        l_t = df[df.columns[0]].astype(bool)
        l_t[l_t==True] = False

        for rule in ruleset:
            l = df[df.columns[0]].astype(bool)
            l[l==False] = True
            for attr, condition in rule.items():
                if condition[0] == '==':
                    l &= df[attr] == condition[1]
                elif condition[0] == '!=':
                    l &= df[attr] != condition[1]
                elif condition[0] == '>=':
                    l &= df[attr] >= condition[1]
                elif condition[0] == '<=':
                    l &= df[attr] <= condition[1]
            l_t |= l
        return np.array(l_t)

    def remove_cases(self, df, ruleset):
        l_t = self.bindings(df, ruleset)
        df = df[~l_t]
        return df

if __name__ == '__main__':
    df = pd.read_pickle('../data/process_data.pkl')
    labels = df.pop('14d')
    df = df.fillna(-999)
    rp = Ripperk()
    rp.fit(df, labels)
    pred = rp.predict(df)
    print(classification_report(labels, pred))
