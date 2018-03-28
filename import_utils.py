import os
path = os.path.join(os.path.abspath('').rsplit('knjk', 1)[0], 'knjk')
import pandas as pd
import numpy as np
import datetime
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse as ssp
from scipy.stats import ks_2samp
import json
import pyecharts
from collections import defaultdict

from . import feature_evaluation
from .operatehdfs import OperateHdfs
from . import feature_evaluation
from . import feature_explore
from . import feature_encoding
from .model_evaluation import xgb_model_evaluation, params
from . import data_utils
from . import visualize
from .psi import Psi
from .rule_learning import Ripperk
from sklearn.metrics import classification_report
from . import rule_learning
from . import facets
from sklearn import metrics  as mr
from sklearn.model_selection import train_test_split
import xgboost as xgb

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

target_cols = ['00d', '01d', '03d', '07d', '14d', '30d']

datapath = 'datasource/'
mtlpath = path + '/mtlsource/'
reportpath = 'reportsource/'

if not os.path.exists(mtlpath):
    os.mkdir(mtlpath)
if not os.path.exists(datapath):
    os.mkdir(datapath)
if not os.path.exists(reportpath):
    os.mkdir(reportpath)

rdata = OperateHdfs()