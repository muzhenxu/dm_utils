#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier#, ExtraTreesClassifier
import xgboost as xgb

out_dir="./"#日志输出 模型及中间固件输出（onehotencoding、scaler） 生成的测评图表输出（auc ks recall 等)
random_state = 2018 #随机种子 （复现用）
max_search = 20 #组合列表中最多搜寻次数（资源消耗、训练速度与最优模型寻找间的平衡）


#训练流程
scoring="roc_auc"#模型评价指标 auc
cv = 5#train 被分割的个数 5 或者 10

estimators={}#期望使用的模型列表
estimators_param={}#期望使用的模型 相关需要search 的参数列表

#所有相关模型 如果需要使用 请先引入相关包（并且要支持fit ，predict 等相关方法）
estimators['rf']=RandomForestClassifier()
estimators_param['rf']={'n_estimators':[1000],
                        'max_depth':[1,3,6,20],
                        'min_samples_split':[60],
                        'min_samples_leaf':[30],
                        'max_features':['auto'],# the same as 'sqrt'
                        'class_weight':[None],
                        'n_jobs':[-1],#线程模式 -1 为使用全部 cup core 资源
                        'random_state':[random_state]# for rebuild
                       }

estimators['ada']=AdaBoostClassifier()
estimators_param['ada']={'n_estimators':[300,1000],
                         'learning_rate':[0.001,0.01,0.03,0.1],
                         'random_state':[random_state]
                         }


estimators['gbdt']=GradientBoostingClassifier()
estimators_param['gbdt']={'n_estimators':[1000],
                          'learning_rate':[0.03,0.1],
                          'max_depth':[1,2,3],
                          'random_state':[random_state]
                          }



estimators['xgb']=xgb.XGBClassifier()
estimators_param['xgb']={'max_depth':[1,3,5],
                        'n_estimators':[1000],
                        'silent':[True],
                        'nthread':[-1],
                        'learning_rate':[1e-2,1e-3],
                        'subsample':[0.8],
                        'gamma':[0],
                        'reg_alpha':[0],#L1
                        'reg_lambda':[1e-3,1e-4],#L2
                        'seed':[random_state]
                        }


class Config:
#配置类
    def __init__(self,df,feature_column_names,label_column_names,other_column_names,positive_label_value=1,output_dir=None):
        '''
        feature_column_names: 特征列名列表 (list)
        label_column_names: target列名列表 (list)
        other_column_names: 其他列名列表 (list)
        positive_label_value: target 列中 正例 的值是多少 (int default = 1)
        output_dir : 输出目录 （str default None)
        '''
        self.df = df
        self.feature_column_names = feature_column_names
        self.label_column_names = label_column_names
        self.other_column_names = other_column_names
        self.positive_label_value = positive_label_value
        self.output_dir = output_dir
        # 下面填充其他需要的配置


    def __str__(self):
        return str(self)
