#!/usr/bin/python
# -*- coding: utf-8 -*-
class Config:
#配置类
    def __init__(self,df,feature_column_names,label_column_names,other_column_names,positive_label_value=1,output_dir=None):
        """
        feature_column_names: 特征列名列表 (list)
        label_column_names: target列名列表 (list)
        other_column_names: 其他列名列表 (list)
        positive_label_value: target 列中 正例 的值是多少 (int default = 1)
        output_dir : 输出目录 （str default None)
        """
        self.df = df
        self.feature_column_names = feature_column_names
        self.label_column_names = label_column_names
        self.other_column_names = other_column_names
        self.positive_label_value = positive_label_value
        self.output_dir = output_dir
        # 下面填充其他需要的配置


    def __str__(self):
        return str(self)
