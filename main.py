import pandas as pd
from timeit import timeit
from catboost import CatBoostRegressor, Pool
import torch
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")

path = 'data/train.csv'

start = timeit()
df_train = pd.read_csv(path)
end = timeit()
print(end-start)
print(df_train.columns.to_list())

constant_columns = [col for col in df_train.columns if df_train[col].nunique() == 1]
print(f'Columns : {constant_columns}, \n Num of Columns : {len(constant_columns)}')


df_train['fullVisitorId'] = df_train['fullVisitorId'].astype(float)
df_train['sessionId'] = df_train['sessionId'].astype(float)

df_train.drop(constant_columns, axis=1, inplace=True)

cat_columns = list(df_train.dtypes[df_train.dtypes == 'object'].reset_index()['index'])
num_columns = num_cols = ["visitNumber", "visitStartTime",]

X = df_train[cat_columns]
y = df_train['totals']


def transform_cat(cat_columns, df_train):
    for col in cat_columns:
        df_train[col] = LabelEncoder().fit_transform(df_train[col])
        return df_train

def xgbRegressor(X_train, y_train, params, cv):

    model = XGBRegressor(params)
    cat_columns = list(X_train.dtypes[X_train.dtypes == 'object'].reset_index()['index'])
    X_train = transform_cat(cat_columns, X_train)
    # evaluate model
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = abs(scores)
    print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

def catboost(X_train, y_train):
    model = CatBoostRegressor(iterations=2,
                              learning_rate=1,
                              depth=2)
    model.fit(X_train, y_train)
    
    print('catboost regressir is fitted')

