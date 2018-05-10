import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pandas as pd

df = pd.read_csv('data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['Class'], 1))
Y = np.array(df['Class'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1],[4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)