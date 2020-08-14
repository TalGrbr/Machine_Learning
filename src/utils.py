import pandas as pd


def get_raw_data():
    return pd.read_csv('../resources/winequality-white.csv', sep=';')


def get_data():
    data = get_raw_data()
    # print(data.describe())
    return data.drop(['quality'], axis=1), data['quality']

