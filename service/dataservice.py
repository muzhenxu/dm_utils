import pandas as pd

class DataService():
    def __init__(self, data_dst):
        self.data_dst = data_dst

    #  目前data_dst直接就是df
    def get_data(self):
        df  = pd.read_pickle(self.data_dst)
        return df