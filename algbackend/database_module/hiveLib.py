import hdfs
import pandas as pd
import numpy as np
import os
import shutil
import subprocess
import re
os.environ.update({'JAVA_HOME': '/usr/local/jdk1.8.0_74'})


class HiveLib:
    def __init__(self, uri='http://two-stream-master-prod-02:50070', user='fan_cx', 
                root='hdfs://nameservicestream/user/hive/warehouse'):
        self.client = hdfs.Client(uri)
        self.user = user
        self.root = root

    
    def _getTableColumns(self, library, table):
        """
        args:
            library: 库名
            table:   表名
        returns:
            columns: 表的列名
        """
        import re
        pattern = re.compile('((^\+---.*)|(^\|.*col_name.*\|.*data_type.*))')

        table = table.lower()
        cmd = f'hive -e "desc {library}.{table}"'
        results = os.popen(cmd).readlines()
        for i, line in enumerate(results):
            if pattern.match(line):
                tmp = results[i:]
                columns = [col.split('|')[1].strip() for col in tmp if not pattern.match(col)]
                return columns

            
    def saveTableToLocal(self, library, table, out_dir='Downloads', out_type='csv', delimiter='\x01', 
                                overwrite_file=False, encoding='utf-8'):
        """
        args:
            library:        库名
            table:          表名
            out_dir:        输出的文件夹
            out_type:       输出文件的类型
            delimiter:      分隔符
            overwrite_file: 是否覆盖输出文件夹下已存在的文件
            encoding:       输出文件的encoding for csv/excel

        returns:
            columns:        该table的df
        """
        if not os.path.exists(out_dir) and out_dir:
            os.mkdir(out_dir)

        if not out_dir:
            out_dir = '.'
        table = table.lower()
        out_path = f'{out_dir}/{table}.{out_type}'

        if os.path.exists(out_path):
            print('INFO  : File Exists!')
            if not overwrite_file:
                try:
                    if out_type == 'csv':
                        df = pd.read_csv(out_path, encoding=encoding)
                    elif out_type == 'xlsx':
                        df = pd.read_excel(out_path, encoding=encoding)
                    elif out_type == 'pkl':
                        df = pd.read_pickle(out_path)
                except:
                    print('ERROR : Something errors.')
                return df
            else:
                print(f'INFO  : Remove Existing .{out_type} File')
                os.remove(out_path)

        cmd = f"hdfs dfs -get {self.root}/{library}.db/{table}  {out_dir}"
        print('INFO  :', cmd)
        results = os.popen(cmd).readlines()
        files_list = os.listdir(f'{out_dir}/{table}')
        columns = self._getTableColumns(library, table)

        df = []
        for file in files_list:
            tmp = pd.read_csv(f'{out_dir}/{table}/{file}', delimiter=delimiter, header=None)
            tmp.columns = columns
            df += [tmp]
        df = pd.concat(df)
        df = df.replace('\\N', np.nan)
        df = df.apply(pd.to_numeric, errors='ignore')
        if out_type == 'csv':
            df.to_csv(out_path, index=False, encoding=encoding)
        elif out_type == 'xlsx':
            df.to_excel(out_path, index=False, encoding=encoding)
        elif out_type == 'pkl':
            df.to_pickle(out_path)
        else:
            print('ERROR : Output file type must be in (csv, xlsx, pkl)')
        rm_path = f'{out_dir}/{table}'
        shutil.rmtree(rm_path)
        return df


    # # 从HDFS文件系统中读取文件，按行返回成list变量
    # def readHDFSFile(self, file_path, delimiter='\x01'):
    #     lines = []
    #     with self.client.read(file_path) as file:
    #         lines = [str(line, 'utf-8').strip().split(delimiter) for line in file]
    #         # for line in file:
    #             # lines.append(str(line, 'utf-8').strip().split(delimiter))
    #     return lines


    # 从我的Hive数据库 dm_fan_cx 里面读取table，存成本地的csv文件，存到datasource下（jupyter文件系统，即hadoop-hd5下）
    def createTableFromLocal(self, library, table, file_path, delimiter='\x01',
                             overwrite_table=False, seq=',',  encoding=None):
        """
        args:
            library:         库名
            table:           表名
            file_path:       输入的文件路径
            delimiter:       分隔符
            overwrite_table: 是否覆盖输出文件夹下已存在的文件
            seq:             如果是csv或者excel的文件分隔符
            encoding:        如果是csv或者excel的编码模式
        returns:
            columns:         None
        """   
        if not os.path.exists(file_path):
            raise Exception(f'INFO  : File {file_path} doesn\'s exist!')
        elif file_path.strip().endswith('.csv'):
            df = pd.read_csv(file_path, sep=delimiter, encoding=encoding)
        elif file_path.strip().endswith('.xlsx'):
            df = pd.read_excel(file_path, sep=delimiter, encoding=encoding)
        elif file_path.strip().endswith('.pkl'):
            df = pd.read_pickle(file_path)
        else:
            raise Exception('INFO  : Input file format must be in (csv, xlsx, pkl).\nCannot read this file!') 
        
        table = table.lower()
        table_name = f'{library}.{self.user}_{table}'
        table_path = f'{self.root}/{library}.db/{self.user}_{table}'

        # 是否重写该 table
        if overwrite_table:
            sql = f'"DROP TABLE IF EXISTS {table_name};"'
            cmd = ['hive', '-e', sql]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(process.stdout.readline, b''):
                print(line)
            process.wait()
            print('INFO  : Droped Existing Table.')

        # 将数据格式转换成我们常用的显示方式
        def chtype(s):
            s = re.sub('[\d]+', '', s)
            if s not in ['int', '']:
                s = 'string'
            if s == 'int':
                s = 'bigint'
            return s

        # 获取每个 column 的数据格式和 column 名
        df_type = df.dtypes.reset_index()
        df_type.columns = ['name', 'data_type']
        df_type['data_type'] = df_type['data_type'].map(lambda s: chtype(str(s)))
        df_type['features'] = df_type.apply(lambda s: '`' + s[0] + '`' + ' ' + s[1], axis=1)
        columns = ', '.join(df_type['features'])

        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ( {columns} ) ROW FORMAT DELIMITED FIELDS TERMINATED BY '{delimiter}' STORED AS TEXTFILE;"
        cmd = ['hive', '-e', f'"{sql}"']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        for line in iter(process.stdout.readline, b''):
            print(line)
        process.wait()

        for i in range(20000):
            temp_file = f'000000_{i}'
            if not os.path.exists(temp_file):
                break
        df.to_csv(temp_file, index=False, header=False, sep=delimiter)

        # sql = f'"LOAD DATA LOCAL INPATH \'{temp_file}\' INTO TABLE {table_name}"'
        # cmd = ['hive', '-e', sql] 
        # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # for line in iter(process.stdout.readline, b''):
        #     print(line)
        # process.wait()
        
        cmd = f'hdfs dfs -put {temp_file} {table_path}'
        print('INFO  :', cmd)
        
        results = os.popen(cmd).readlines()
        for line in results:
            print('INFO  :', line)        
        os.remove(temp_file)

        
if __name__ == '__main__':
    print('test')
    hive_lib = HiveLib()
