from utils import get_extension
import pandas as pd
import numpy as np
from dateutil.parser import parse
from collections import namedtuple


class Question:
    def __init__(self, question, default_value=None, right_answers=None):
        self.right_answers = right_answers
        self.question = question
        self.default_value = default_value
        self.default_value_quest = '' if default_value is None else f'[{self.default_value}]'

    def ask(self):
        good_response = False
        while not good_response:
            response = input(f'{self.question}{self.default_value_quest} ')
            if not response:
                response = self.default_value
            good_response = (not self.right_answers) or response in self.right_answers
            if self.right_answers and not good_response:
                print(f'Allowed answers to this questions are {self.right_answers}\n')
        return response


ColsInfo = namedtuple('ColsInfo', ['catcols', 'quacols', 'datecols', 'idcol', 'targetvar'])


def find_train_test():
    where_train = Question("Where is your train dataset under project root?", 'data/raw/train.csv')
    where_test = Question("Where is your test dataset under project root?", 'data/raw/test.csv')
    train_path = where_train.ask()
    test_path = where_test.ask()
    is_csv = get_extension(train_path) == 'csv'
    if is_csv:
        what_sep = Question('Specify separator into csv file:', ',')
        csv_sep = what_sep.ask()
        train, test = pd.read_csv(train_path, csv_sep), pd.read_csv(test_path, csv_sep)
    else:
        load_f = getattr(pd, 'read_' + get_extension(train_path))
        train = load_f(train_path)
        load_f = getattr(pd, 'read_' + get_extension(test_path))
        test = load_f(test_path)
    return train, test


def isincreasing(values):
    return all((values[i] + 1) == values[i + 1] for i in range(len(values) - 1))


def check_type_cols(test):
    catcols, quacols, datecols = [], [], []
    has_id = Question('Has this dataset an id column?', 'y', ['y', 'n']).ask() == 'y'
    id_found = False
    id_col = None
    for col in test.columns:
        if test[col].dtype == np.object:
            if has_id and not id_found and 'id' in col.lower():
                quest_is_id = Question(f"Is {col} the id column?", 'y', ['y', 'n'])
                is_id = quest_is_id.ask() == 'y'
                if is_id:
                    id_col = col
                    id_found = True
                    continue
            try:
                date = parse(test[col].iloc[0])
                datecols.append(col)
            except:
                catcols.append(col)
        else:
            values = sorted(set(test[col].values))
            if values[0] in [0, 1] and isincreasing(values):
                quest_is_cat = Question(f"Is {col} a categorical column?", 'y', ['y', 'n'])
                is_cat = quest_is_cat.ask() == 'y'
                if is_cat:
                    catcols.append(col)
                else:
                    quacols.append(col)
            else:
                quacols.append(col)
    if has_id and not id_found:
        specify_id = Question(f'Please specify manually the id column from these: {test.columns}',
                              right_answers=test.columns.tolist())
        id_col = specify_id.ask()
        if id_col in catcols: catcols.remove(id_col)
        if id_col in quacols: quacols.remove(id_col)
    return (catcols, quacols, datecols, id_col)


def write_into_dataset_py(train_shape, test_shape, colsinfo: ColsInfo):
    train_rows, train_cols = train_shape
    test_rows, test_cols = test_shape
    text = f"""NROWS_TRAIN = {train_rows}
NCOLS_TRAIN = {train_cols}

NROWS_TEST = {test_rows}
NCOLS_TEST = {test_cols}

TARGETVAR = {repr(colsinfo.targetvar)}
IDCOL = {repr(colsinfo.idcol)}

CATEGORICAL_COLS = {colsinfo.catcols}
QUANTITATIVE_COLS = {colsinfo.quacols}
DATES_COLS = {colsinfo.datecols}"""
    with open('constants/dataset.py', mode='w') as f:
        f.write(text)


def main():
    train, test = find_train_test()
    if train.shape[1] - test.shape[1] - 1 != 0:
        print(f"Warning: #columns difference between train and test is {train.shape[1] - test.shape[1]}")
    targetvar = train.columns.difference(test.columns)[0]
    cols_metadata = check_type_cols(test)
    cols_metadata = ColsInfo(*cols_metadata, targetvar)
    write_into_dataset_py(train.shape, test.shape, cols_metadata)
