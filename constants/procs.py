from multiprocessing import Pool, cpu_count
from pandas import DataFrame
from typing import Callable, List, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .dataset import TARGETVAR, NROWS_TRAIN, CATEGORICAL_COLS_RAW
from .paths import PRJ_ROOT, SCORE_FILE
from operator import itemgetter
from math import log
from scipy.stats import chi2_contingency
from datetime import datetime
from functools import partial
import lightgbm as lgb
from sklearn.model_selection import validation_curve
import json


def parallel_map_df(func: Callable[[DataFrame, Any], DataFrame], num_cores=cpu_count()):
    """
    Python decorator that must be wrapped to a function that receive as first input a pandas Dataframe.
    Apply a trasformation on all the dataframe in parallel using the multiprocessing module

    Example:
        @parallel_map_df
        def sum_of_cols(df):
            df['c'] = df['a'] + df['b']
            return df


        @parallel_map_df(num_cores=16)
        def sum_of_cols(df):
            df['c'] = df['a'] + df['b']
            return df
    """

    def wrapper(df: DataFrame, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        partitions = np.linspace(0, len(df) - 1, num_cores + 1).round().astype('int')
        df_split = [df.iloc[partitions[i]:partitions[i + 1]] for i in range(len(partitions) - 1)]
        pool = Pool(num_cores)
        df = pd.concat(pool.map(partial_func, df_split))
        pool.close()
        pool.join()
        return df

    return wrapper


def _take_first_k(seq: np.array, k, keyf=lambda x: x) -> np.array:
    n = len(seq)
    if k > log(n):
        return sorted(seq, key=keyf)[:k]
    first_k = []
    for _ in range(k):
        max_el = max(seq, key=keyf)
        first_k.append(max_el)
        seq[max_el[0]][1] = -1  # to avoid remotion
    return first_k


def adversial_validation(train: DataFrame, test: DataFrame, perc_val=0.3, model=RandomForestClassifier()) -> DataFrame:
    """
    Do an adversial validation on the input datasets. Basically the adversial validation consists on finding the most similar
    validation set to the test test. You can find more info here: http://manishbarnwal.com/blog/2017/02/15/introduction_to_adversarial_validation/
    Attention: the columns on train and test must be in a numeric form
    """
    train_orig = train
    train = train.drop(TARGETVAR, axis=1)
    if TARGETVAR in test.columns:
        test = test.drop(TARGETVAR, axis=1)
    train['__target'] = 0
    test['__target'] = 1
    dataset = pd.concat([train, test])
    X, y = dataset.drop('__target', axis=1), dataset['__target']
    model.fit(X, y)
    del train['__target']
    del test['__target']
    p = list(enumerate(map(itemgetter(1), model.predict_proba(train))))
    k = int(round(perc_val * NROWS_TRAIN))
    higher_proba = _take_first_k(p, k, itemgetter(1))
    return train_orig.iloc[list(map(itemgetter(0), higher_proba))]


def _cramerv(col1: np.array, col2: np.array):
    confusion_matrix = pd.crosstab(col1, col2)
    chi2, p_value = chi2_contingency(confusion_matrix)[0:2]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if not phi2corr == 0 else 0, p_value


def cramerv(dataset: DataFrame, cols: List[str] = CATEGORICAL_COLS_RAW) -> np.matrix:
    """
    Apply the cramerv test on each pair of columns. The cramerv test is used to
    measure in [0,1] the correlation between categorical variables
    :param dataset: Dataframe where the correlation is measured
    :param cols: the columns of dataset where measure the correlation
    :return: the correlation matrix of cramerv on the specified columns of dataset
    """
    matr = np.matrix([[0] * len(cols)] * len(cols), dtype='float')
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            if i == j:
                matr[i, j] = 1
            else:
                corr, pvalue = _cramerv(dataset[cols[i]], dataset[cols[j]])
                matr[i, j] = corr
                matr[j, i] = corr
    return matr


def save_predictions(y_pred, index=None):
    """
    Save in predictions folder your model output using the current datetime into the filename
    :param y_pred: predictions
    :param index: List of indexes to put before y_pred
    :return: None
    """
    prediction_fold = PRJ_ROOT / 'outputs'
    if not prediction_fold.exists():
        prediction_fold.mkdir()
    currtime = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    res = pd.DataFrame({'index': index, 'ypred': p}) if index else pd.DataFrame(y_pred)
    res.to_csv(prediction_fold / 'prediction_' + currtime + '.csv', index=False, header=False)


def save_score(model: str, params: Dict[str, Any], scores: Dict[str, Any], comment='',
               date: Union[str, datetime] = 'now'):
    data = {'model': model, 'comment': comment,
            **{f'param_{param}': param_value for param, param_value in params.items()},
            **{f'score_{score}': score_value for score, score_value in scores.items()}}
    if date == 'now':
        date = datetime.now()
    if date:
        data['date'] = date
    with open(SCORE_FILE, mode='r') as score_f:
        try:
            old_scores = json.load(score_f)
        except json.JSONDecodeError:
            old_scores = []
    with open(SCORE_FILE, mode='w+') as score_f:
        json.dump(old_scores + [data], score_f)


def load_scores() -> DataFrame:
    return pd.read_json(SCORE_FILE)

from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def fill_na_numeric(df, column):
    model = RandomForestRegressor(n_jobs=-1)
    return _fill_na(df, column, model)

def fill_na_category(df, column):
    model = RandomForestClassifier(n_jobs=-1)
    return _fill_na(df, column, model)

def _fill_na(df, column, model):
    na_rows = df[column].isna()
    train, test = df[~na_rows], df[na_rows].drop(columns=column)
    model.fit(train.drop(columns=column).values, train[column].values)
    del train
    new_values = model.predict(test.values)
    del model
    test[column] = new_values
    df.loc[test.index, column] = test[column]
    del test
    return df

def cat_to_int(df):
    strcols = list(df.select_dtypes(include=['object', 'category']).columns)
    for strcol in strcols:
        df[strcol] = pd.to_numeric(df[strcol].astype('category').cat.codes, downcast='integer')
    return df

def fill_quantitative_with_mean(df, quant_cols):
    df[quant_cols] = df[quant_cols].fillna(df[quant_cols].mean())
    return df

def fill_na_categories(df, category_cols_na):
    quantitative_na_cols = list(df.loc[:, df.isna().any()].select_dtypes(exclude=['category', 'object', 'bool']).columns)
    for category_col_na in category_cols_na:
        tmp = df.copy()
        tmp = fill_quantitative_with_mean(tmp, quantitative_na_cols)
        tmp = cat_to_int(tmp)
        tmp[category_col_na] = tmp[category_col_na].replace(-1, np.NaN)
        tmp = fill_na_category(tmp, category_col_na)
        del train_test[category_col_na]
        df[category_col_na] = tmp[category_col_na]
    return df

def validate_param(param_name, param_values, train, model=lgb.LGBMRegressor(n_jobs=-1), scoring='accuracy'):
    X, y = train.drop(columns=TARGETVAR).values, train[TARGETVAR].values
    scorename = scoring if isinstance(scoring, str) else scoring.__name__
    train_scores, valid_scores = validation_curve(model, X, y, param_name, param_values, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.title("Validation Curve with "+model.__class__.__name__)
    plt.xlabel(param_name)
    plt.ylabel(scorename)
    plt.semilogx(param_values, train_scores_mean, label="Training score", color='#78A8F3')
    plt.fill_between(param_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="#78A8F3")
    plt.semilogx(param_values, valid_scores_mean, label="Cross-validation score",
                 color="#F37878")
    plt.fill_between(param_values, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.2,
                     color="#F37878")
    plt.legend(loc="best")
    plt.show()
    return train_scores, valid_scores

__all__ = ['parallel_map_df', 'adversial_validation', 'save_predictions', 'save_score', 'load_scores', 'cramerv', 'cat_to_int', 'fill_na_category', 'fill_na_numeric', 'validate_param']
