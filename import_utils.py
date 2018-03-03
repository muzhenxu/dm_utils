import sys
import os
path = os.path.join(os.path.abspath('').rsplit('knjk', 1)[0], 'knjk')
sys.path.append(path)
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

from utils import feature_evaluation
from utils.operatehdfs import OperateHdfs
from utils import feature_evaluation
from utils import feature_explore
from utils.model_evaluation import xgb_model_evaluation
from utils import data_utils
from utils import visualize
from utils.psi import Psi
from utils.rule_learning import Ripperk
from sklearn.metrics import classification_report
from utils import rule_learning

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