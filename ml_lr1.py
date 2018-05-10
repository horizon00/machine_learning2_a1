import pandas as pd
import math
import datetime
import numpy as np
import quandl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


import matplotlib.pyplot as plt
from matplotlib import style

import pickle

style.use('ggplot')
df = quandl.get('FINRA/FNSQ_GOOGL')
df = df[['ShortVolume', 'ShortExemptVolume', 'TotalVolume']]
df['PCT'] = df['ShortVolume']/df['TotalVolume']*100.0
df['ShortExemptVolume_PCT'] = df['ShortExemptVolume']/df['TotalVolume']*100.0
df = df[['ShortVolume', 'ShortExemptVolume', 'TotalVolume', 'PCT', 'ShortExemptVolume']]
forecast_col = 'ShortVolume'
df.fillna(-99999, inplace=True)
df.dropna(inplace=True)
forecast_out = int(math.ceil(1.6*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)




X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out+1:]

Y = np.array(df['label'])
Y = np.array(df['label'])


X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size=0.01)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, Y_train)
with open('LinearRegression.pickle','wb') as f:
    pickle.dump(clf, f)

pickel_in = open('LinearRegression.pickle', 'rb')
clf = pickle.load(pickel_in)


accuracy = clf.score(X_test, Y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400

next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc['next_date'] = [np.nan for _ in range(len(df.columns)-1)]+[i]

    df['ShortExemptVolume'].plot
    df['forecast'].plot
    plt.legend(loc=4)
    plt.xlabel('date')
    plt.ylabel('price')
    plt.show()