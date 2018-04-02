import string
import hashlib
import random
import os
import pandas as pd
import json
import numpy as np

def md5(s):
    m = hashlib.md5()
    m.update(str(s).encode())
    return m.hexdigest()

def passwd():
    salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    print(salt)
    return salt

def merge_report(reportpath='reportsource/'):
    writer = pd.ExcelWriter(reportpath + 'report.xlsx')
    for f in os.listdir(reportpath):
        if f.endswith('pkl'):
            sheet = pd.read_pickle(reportpath + f)
            sheet.to_excel(writer, sheet_name=f[:-4], encoding='gbk')
    writer.save()


def get_model_data(s, method='response',  key='score'):
    if method == 'response':
        try:
            return json.loads(s)['data'][key]
        except:
            return np.nan
    elif method == 'request':
        try:
            return json.loads(s)[key]
        except:
            return np.nan
