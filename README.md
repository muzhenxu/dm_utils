### \*此项目仅用于公司内部使用\*
---

>此项目使用于 获取数据 后续的分析、处理、转化、建模、评测等。

>初始数据格式 dataframe 包含3个部分 feature_columns 特征列，label_columns （目标列 ），other_columns(其他列 不进入模型 例如贷后 其他数据 催收情况等）

>用于固化获取数据后模型训练的全部流程，减少人肉误差。


### 准备工作：
---

>DataFrame  用于本次探索的数据 包含特征与目标列 的 集合。


### 模块简介：
---

| 模块           | 功能           | 解释          |文件|
| ------------- |:-----------------------------:|:-------------:|:-------------:|
| `config`      | 配置信息 |预设后续流程的初始化配置 便于统一更改 （包括提供 特征列，目标列 及其他后续流程的初始配置） |./config.py|
| `analyze`     | 互信息、覆盖率、分布、iv、auc、ks、排序 | 可用于模型前数据分析与建模后test数据集分析 |./analyze/\*|
| `transform`   | 目录类变量扩展（one_hot_encoding）,缺失值填充（knn,mean,frequency)     | 对基础数据做转换，转化过程对象记录为外部文件 |./transform/\*|
| `anomaly-detection` | 异常检测 |根据高斯分布计算每行特征概率并返回p值列表 |./anomaly_detection/\*|
| `hold-out`      | 留出法 | 根据配置（种子，是否乱序，留出比例）对原datafram 进行划分，返回train 与test 集合 |./hold_out/hold_out.py|
| `train-models`  | 训练模型 | 根据配置（cv 个数，模型列表，各模型参数搜寻列表，评价指标）对留出法中的train 数据集进行训练，得到各类模型中最优模型 |./train_models/\*|
| `tutorial`  | 其他 | 包含各方法使用样例及所需数据文件（上传时注意大小，最大文件200KB) |./tutorial/\*|


### 模块详细介绍：
---

1.留出法    
   > ** 教程： ./tutorial/hold_out_test.py **
