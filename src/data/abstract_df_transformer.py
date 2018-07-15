from abc import ABC, abstractmethod, ABCMeta
from path import Path
from constants.dataset import IDCOL
from data.train_test_loader import TrainTestLoader


class AbstractDFTransformer(TrainTestLoader, ABC):
    def __init__(self):
        self.drop_id = True
        self.first_train = True

    def fit_transform(self, train: Path = None, test: Path = None):
        train, test = self.load_train_and_test(train, test)
        if self.drop_id:
            train_id, test_id = train[IDCOL], test[IDCOL]
            train.drop(columns=IDCOL, inplace=True)
            test.drop(columns=IDCOL, inplace=True)
        if self.first_train:
            train = self.transform_train(train)
            test = self.transform_test(test)
        else:
            test = self.transform_test(test)
            train = self.transform_train(train)
        if self.drop_id:
            train[IDCOL] = train_id
            test[IDCOL] = test_id
        return train, test

    @abstractmethod
    def transform_train(self, train):
        pass

    @abstractmethod
    def transform_test(self, test):
        pass
