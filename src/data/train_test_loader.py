from abc import abstractmethod, ABC
from argparse import ArgumentParser

import pandas as pd
from path import Path

from constants.dataset import TARGETVAR
from utils import fixed_argv, get_extension, get_filename


class TrainTestLoader(ABC):

    @abstractmethod
    def fit_transform(self, train, test):
        pass

    def parseargs(self, train_none, test_none):
        print(train_none)
        argparser = ArgumentParser()
        if train_none:
            argparser.add_argument('--train', type=str, default='raw/train.csv')
        if test_none:
            argparser.add_argument('--test', type=str, default='raw/test.csv')
        args = argparser.parse_args(fixed_argv(__file__))
        self.train_name = get_filename(args.train)
        self.test_name = get_filename(args.test)
        return args

    def load_train_test_then_concat(self, trainpath: Path, testpath):
        train, test = self.load_train_and_test(trainpath, testpath)
        y = train[TARGETVAR].values
        train_len = len(train)
        train.drop(columns=TARGETVAR, inplace=True)
        train_test = pd.concat([train, test], copy=False)
        train_test.reset_index(inplace=True, drop=True)
        del train, test
        return train_test, y, train_len

    def load_train_and_test(self, train, test):
        if isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame):
            return train, test
        trainextension, testextension = get_extension(train.basename()), get_extension(test.basename())
        trainloadf, testloadf = getattr(pd, 'read_' + trainextension), getattr(pd, 'read_' + testextension)
        train: pd.DataFrame = trainloadf(train)
        test: pd.DataFrame = testloadf(test)
        return train, test


