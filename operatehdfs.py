import hdfs
import pandas as pd
import os
import subprocess
import re


class OperateHdfs(object):
    def __init__(self, user='dm_xu_h', ip_port='http://hadoop-hd1:50070', root='/user/hive/warehouse'):
        self.client = hdfs.Client(ip_port)
        self.user = user
        self.root = root

    def getColumn(self, table, use_table=True):
        table_name = '%s.%s' % (self.user, table)
        if use_table:
            columns = os.popen('hive -e "desc %s"' % table_name).readlines()
            columns = [c.split('\t')[0].strip() for c in columns]
            return columns

    def readHdfsFile(self, file_path):
        l = []
        with self.client.read(file_path) as f:
            for line in f:
                l.append(str(line, 'utf-8').strip().split('\x01'))
        return l

    def readHdfsTable(self, table, drop_csv=False):
        table_path = '%s/%s.db/%s' % (self.root, self.user, table)
        l = []

        if not os.path.exists('datasource'):
            os.mkdir('datasource')
        csv_path = 'datasource/%s.csv' % os.path.basename(table_path)
        if os.path.exists(csv_path):
            print('file exists!')
            if not drop_csv:
                df = pd.read_csv(csv_path)
                return df
            else:
                print('remove existing csv')
                os.remove(csv_path)

            file_list = self.client.list(table_path)
            for f in file_list:
                l.extend(self.readHdfsFile(os.path.join(table_path, f)))

            columns = self.getColumn(table)
            df = pd.DataFrame(l, columns=columns)
            df.to_csv(csv_path, index=False)
            return df

    def readfromcsv(self, table, drop_csv=False):
        table_name = '%s.%s' % (self.user, table)
        table_path = '%s/%s.db/%s' % (self.root, self.user, table)
        if not os.path.exists('datasource'):
            os.mkdir('datasource')

        csv_path = 'datasource/%s.csv' % os.path.basename(table_path)
        if os.path.exists(csv_path):
            print('file exists!')
            if not drop_csv:
                df = pd.read_csv(csv_path)
                return df
            else:
                print('remove existing csv')
                os.remove(csv_path)

        columns = os.popen('hive -e "desc %s"' % table_name).readlines()
        columns = [c.split('\t')[0].strip() for c in columns]

        # cmd = 'hive -e "select * from %s" >> %s' % (table_name, csv_path)
        cmd = ['hive', '-e', '"select * from %s"' % table_name]
        with open(csv_path, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)
        for line in iter(process.stderr.readline, b''):
            print(line)
        process.wait()

        df = pd.read_csv(csv_path, delimiter='\t', names=columns)
        df.to_csv(csv_path, index=False)

        return df

    def savetohdfs(self, table, file, drop_table=False, sep=',', encoding=None):
        table_name = '%s.%s' % (self.user, table)
        if file.split('.')[-1] == 'csv':
            df = pd.read_csv(file, sep=sep, encoding=encoding)
        elif file.split('.')[-1] == 'pkl':
            df = pd.read_pickle(file)
        else:
            return "can't read this file !"

        if drop_table:
            cmd = ['hive', '-e', 'DROP TABLE IF EXISTS %s' % table_name]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(process.stdout.readline, b''):
                print(line)
            process.wait()

        df_type = df.dtypes.reset_index()
        df_type.columns = ['name', 'data_type']

        def chtype(s):
            s = re.sub('[\d]+', '', s)
            if s not in ['int', 'float']:
                s = 'string'
            return s

        df_type.data_type = df_type.data_type.map(lambda s: chtype(str(s)))
        df_type['features'] = df_type.apply(lambda s: '`' + s[0] + '`' + ' ' + s[1], axis=1)
        columns = ','.join(df_type['features'])
        sql = "create table %s (%s) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' STORED AS TEXTFILE;" % (
            table_name, columns)
        cmd = ['hive', '-e', '"%s"' % sql]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            print(line)
        process.wait()

        for i in range(10000, 20000):
            temp_file = 'temp_save_to_hdfs_%s.csv' % i
            if not os.path.exists(temp_file):
                break
        df.to_csv(temp_file, index=False, header=False, sep='\t')

        cmd = ['hive', '-e', '"load data local inpath \'%s\' into table %s;"' % (temp_file, table_name)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            print(line)
        process.wait()
        os.remove(temp_file)

if __name__ == '__main__':
    hdata = OperateHdfs(user='dm_xu_h')
    df = hdata.readfromcsv('tablename')
    hdata.savetohdfs(table='newtablename', file='name.csv')