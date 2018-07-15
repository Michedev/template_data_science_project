import pandas as pd
import numpy as np
from abstract_traintest_transformer import AbstractTrainTestTransformer
from random import sample
from constants.paths import DATA_PROCESSING


class StatsExtractor(AbstractTrainTestTransformer):

    def __init__(self):
        self.drop_id = True

    def transform(self, train_test):
        orig_columns = train_test.columns.tolist()
        train_test = self.extract_stats(orig_columns, train_test)
        train_test = self.extract_not_zero_stats(orig_columns, train_test)
        train_test = self.extract_moving_mean(orig_columns, train_test)
        train_test = train_test.loc[:, ~((train_test == np.infty).any())]
        train_test = train_test.loc[:, ~((train_test > 2 ** 63).any())]
        train_test = train_test.loc[:, ~(train_test.isna().any())]
        return train_test

    def extract_moving_mean(self, orig_columns, train_test):
        moving_mean_path = DATA_PROCESSING / 'moving_mean.feather'
        if moving_mean_path.exists():
            moving_mean_df = pd.read_feather(moving_mean_path)
        else:
            moving_mean_df = pd.DataFrame()
            for col in orig_columns + ['_mean']:
                moving_mean_df[f'_moving_mean_{col}'] = train_test[[col]].rolling(2, min_periods=1).sum()
                moving_mean_df[f'_log_moving_mean_{col}'] = np.log1p(train_test[f'_moving_mean_{col}'])
            moving_mean_df.to_feather(moving_mean_path)
        train_test = pd.concat([train_test, moving_mean_df], axis=1, copy=False)
        return train_test

    def extract_not_zero_stats(self, orig_columns, train_test):
        stats_not_zero_path = DATA_PROCESSING / 'stats_not_zero_df.feather'
        if stats_not_zero_path.exists():
            stats_not_zero_df = pd.read_feather(stats_not_zero_path)
        else:
            stats_not_zero_df = pd.DataFrame()
            df_non_zero = [train_test.loc[:, orig_columns].iloc[i, :]
                           [train_test.loc[:, orig_columns].iloc[i, :] != 0]
                           for i in range(len(train_test))]
            for stat in ['mean', 'median', 'std', 'kurtosis', 'min', 'mode']:
                newcolname = f'_{stat}_not_zero'
                stats_not_zero_df[newcolname] = [getattr(row_non_zero, stat)() for row_non_zero in df_non_zero]
                stats_not_zero_df[newcolname] = stats_not_zero_df[newcolname].fillna(0)
                stats_not_zero_df[f'_log_{newcolname}'] = np.log1p(stats_not_zero_df[newcolname])
                stats_not_zero_df[f'_square_{newcolname}'] = (stats_not_zero_df[newcolname]) ** 2
                stats_not_zero_df[f'_inverse_{newcolname}'] = 1 / stats_not_zero_df[newcolname]
            stats_not_zero_df.to_feather(stats_not_zero_path)
        train_test = pd.concat([train_test, stats_not_zero_df], axis=1, copy=False)
        return train_test

    def extract_stats(self, orig_columns, train_test):
        stats_path = DATA_PROCESSING / 'stats_df.feather'
        if stats_path.exists():
            statsdf = pd.read_feather(stats_path)
        else:
            statsdf = pd.DataFrame()
            for stat in ['mean', 'var', 'std', 'max', 'min', 'kurtosis']:
                statsdf['_' + stat] = getattr(train_test[orig_columns], stat)(axis=1)
                statsdf['_log_' + stat] = np.log1p(statsdf['_' + stat])
                statsdf['_square_' + stat] = (statsdf['_' + stat]) ** 2
                statsdf['_inverse_' + stat] = 1 / (statsdf['_' + stat])
            statsdf.to_feather(stats_path)
        train_test = pd.concat([train_test, statsdf], axis=1, copy=False)
        return train_test
