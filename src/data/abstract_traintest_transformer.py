import pandas as pd
from abc import abstractmethod, ABCMeta, ABC
from path import Path
from train_test_loader import TrainTestLoader
from constants.dataset import IDCOL, TARGETVAR
from constants.paths import DATA
from typing import Union


class AbstractTrainTestTransformer(TrainTestLoader, ABC):
    drop_id = True

    def fit_transform(self, train: Union[Path, str, pd.DataFrame] = None,
                      test: Union[Path, str, pd.DataFrame] = None):
        if train is None or test is None:
            args = self.parseargs(train is None, test is None)
            if train is None: train = DATA / args.train
            if test is None: test = DATA / args.test
        train_test, self.y, self.train_len = self.load_train_test_then_concat(train, test)

        if self.drop_id and IDCOL in train_test.columns:
            id_train_test = train_test[IDCOL].values
            train_test.drop(columns=IDCOL, inplace=True)
        train_test = self.transform(train_test)
        if self.drop_id:
            train_test[IDCOL] = id_train_test
        train, test = self.split_train_test(train_test, self.y, self.train_len)
        return train, test

    @abstractmethod
    def transform(self, train_test):
        pass

    def split_train_test(self, train_test, y, train_len):
        train, test = train_test.iloc[:train_len], train_test.iloc[train_len:]
        train[TARGETVAR] = y
        return train, test
