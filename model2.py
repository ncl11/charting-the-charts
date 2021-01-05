from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def train_model():
    """alternate lin reg model that can be used instead of random forest, both perform well"""
    df = pd.read_csv('3_mo_weekly.csv', sep='\t')

    train = df[df['Date'] <= '2020-10-18']
    train = train.drop(['Date', 'Release', 'Y'],axis=1)
    y_train = train['Week + 1']
    X_train = train.drop('Week + 1', axis = 1)

    test = df[df['Date'] > '2020-10-18']
    test = test[test['Week + 1'] > 0]
    test = test.drop(['Date', 'Release', 'Y'],axis=1)
    y_test = test['Week + 1']
    X_test = test.drop('Week + 1', axis=1)

    alpha = 2**46
    linreg = Ridge(alpha = alpha, fit_intercept=True)
    linreg.fit(X_train, y_train)

    return linreg