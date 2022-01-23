import pandas as pd
import tsfresh as tf
from utils.data import data_loader


class FeatureExtractor:
    def __init__(self, dataset='Haleh'):
        self.train_np, self.test_np = data_loader(dataset)
        self.train_df, self.test_df = self.prepare_data()

    def prepare_data(self):
        x_train = self.train_np[0]
        x_test = self.test_np[0]

        def _transform_to_df(data):
            df = pd.DataFrame()
            for sample_id in range(data.shape[0]):
                time_series = data[sample_id]
                data_list = [[value, sample_id] for value in time_series]
                df = df.append(data_list, ignore_index=True)
            df.columns = ['value', 'id']
            return df

        df_train = _transform_to_df(x_train)
        df_test = _transform_to_df(x_test)
        return df_train, df_test

    def extract(self):
        tf_train = tf.extract_features(self.train_df, column_id='id')
        tf_test = tf.extract_features(self.test_df, column_id='id')
        return tf_train, tf_test
