import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('diamonds.csv')
dataset.drop('Unnamed: 0', axis = 1, inplace = True)
X = dataset.drop('price', axis = 1)
y = dataset['price']
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[1,2,3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)
X = X[:, 1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
