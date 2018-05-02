import json
import os
from .spliter import data_spliter
from .transformer import FeatureTransformer
from .dataservice import DataService
from .trainer import model_cv_trainer
from .. import config

metadata_path = config.metadata_path


class TrainService():
    def __init__(self, dst, data_dst):
        self.dst = dst
        self.data_dst = data_dst
        self.metadata = self.get_metadata(dst)

    def fit(self):
        DS = DataService(self.data_dst)
        df = DS.get_data()

        train, train_y, test, test_y = data_spliter(df, **self.metadata['spliter'])

        self.enc = FeatureTransformer(**self.metadata['transformer'])
        self.enc.fit(train, train_y)
        train = self.enc.transform(train)
        test = self.enc.transform(test)

        self.clf, df_cv, df_test, df_train = model_cv_trainer(train, train_y, test, test_y, **self.metadata['trainer'])
        return df_cv, df_test, df_train

    def predict(self, df):
        df = self.enc.transform(df)
        pred = self.clf.predict(df)
        return pred

    # TODO: 目前采用文件系统并写死
    @staticmethod
    def get_metadata(dst):
        metadata = json.load(open(os.path.join(metadata_path, 'algchain_metadata'), 'r'))
        metadata = metadata[dst]
        return metadata
