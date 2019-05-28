# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:46:33 2019

@author: Reyad
"""


# this file has been made according to video tutorial of
# machine learning a2z from udemy
# but now since scikit-learn has been changed and became more modern, you should
# consult 'data_preproc_new_style.py' file for data preprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# load data
DATA_PATH = "data/Data.csv"

dataset = pd.read_csv(DATA_PATH)
# seperating independent and dependent variables
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
# and ... more lines
# the reason i'm changing the code is:
# sklearn.preprocessing.Imputer is deprecated

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") # watch we didn't use "axis" parameter.
# of Imputer. "axis" parameter is removed from SimpleImputer. So, SimpleImputer always calculates
# mean of provided columns values.
X.iloc[:, 1:3] = imputer.fit_transform(X.iloc[:, 1:3])


# encoding categorical data
# we have to encode text data into neumerical data
# as the libraries only work with neumerical values
# so, it's good to transform the data
from sklearn.preprocessing import LabelEncoder
# encoding X['Country']
labelEncoder_X = LabelEncoder()
X.iloc[:, 0]  = labelEncoder_X.fit_transform(X.iloc[:, 0])
# one_hot_encoding as no country has higher priority
# cause the machine learning algos set a higher priority for country=2 that country=0 or 1
# so one_hot_encoding
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
oneHotEncoder = oneHotEncoder.fit(X)
X = oneHotEncoder.transform(X).toarray()
# encoding output y['Purchased']
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)


# train and test data set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# feature scaling for bring all features in same importance level
# like feature 'salary' range is 32000 <= salary <= 200000
# but feature 'age' has range 22 <= age <= 49
# as many machine learning algos depend on euclidean distance
# salary gets much attention than age
# so age does not have any impact in the output
# that's why every feature is brought into -1 to 1 values or like so in a small range
# popular two known methods are:
# 1. standardisation [eqn: x_stan = (x - mean(x)) / standard_deviation(x)]
# 2. normalisation [eqn: X_norm = (x - min(x)) / (max(x) - min(x))]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
