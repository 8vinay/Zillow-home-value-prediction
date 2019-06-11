import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.3.0-posix-seh-rt_v5-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from sklearn.cross_validation import train_test_split

train_data = pd.read_csv('train_2016_v2.csv')
properties = pd.read_csv('properties_2016.csv',low_memory=False)
sample_sol = pd.read_csv('sample_submission.csv')

for c, d in zip(properties.columns, properties.dtypes):
	if d == np.float64:
		properties[c] = properties[c].astype(np.float32)


df_train = train_data.merge(properties, on='parcelid',how='left')
df_train.fillna(0)

x_training = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_training = df_train['logerror'].values

print(x_training.shape, y_training.shape)

train_columns = x_training.columns

for c in x_training.dtypes[x_training.dtypes == object].index.values:
    x_training[c] = (x_training[c] == True)

del df_train; gc.collect()


x_training, x_validation, y_training, y_validation = train_test_split(x_training, y_training, test_size=0.3, random_state=100)

d_train = xgb.DMatrix(x_training, label=y_training)
d_valid = xgb.DMatrix(x_validation, label=y_validation)

del x_training, x_validation; gc.collect()

print('Training ...')

parameters = {}
parameters['silent'] = 1
parameters['eta'] = 0.02
parameters['objective'] = 'reg:linear'
parameters['eval_metric'] = 'mae'
parameters['max_depth'] = 4

watchlist = [(d_train, 'training'), (d_valid, 'validation')]
model = xgb.train(parameters, d_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=10)

del d_train, d_valid


sample_sol['parcelid'] = sample_sol['ParcelId']
df_test = sample_sol.merge(properties, on='parcelid', how='left')

del properties; gc.collect()

x_testing = df_test[train_columns]
for c in x_testing.dtypes[x_testing.dtypes == object].index.values:
    x_testing[c] = (x_testing[c] == True)

del df_test, sample_sol; gc.collect()

d_test = xgb.DMatrix(x_testing)

del x_testing; gc.collect()


p_test = model.predict(d_test)

del d_test; gc.collect()

submission = pd.read_csv('sample_submission.csv')
for c in submission.columns[submission.columns != 'ParcelId']:
    submission[c] = p_test

print('Writing to csv ...')
submission.to_csv('xgb.csv', index=False, float_format='%.4f')
