import sys
from copy import deepcopy
import pandas as pd


def fixed_argv(*args):
    toparse = deepcopy(sys.argv)
    if toparse[0] == 'run.py':
        return toparse[2:]
    return toparse


def get_filename(filepath: str):
    return '.'.join(filepath.split('/')[-1].split('.')[0:-1])


def get_extension(filepath: str):
    return filepath.split('.')[-1]


def downcast_float(df):
    for col in df.select_dtypes('float').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def downcast_int(df):
    for col in df.select_dtypes('int').columns:
        df[col] = pd.to_numeric(df[col], downcast='signed')
    return df
