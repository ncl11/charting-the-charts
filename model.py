from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def train_model():
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

    rs=22
    max_depth = 16
    max_features = .6
    rf_reg = RandomForestRegressor(max_depth=max_depth, max_features=max_features, random_state=rs)
    rf_reg.fit(X_train, y_train)

    return rf_reg