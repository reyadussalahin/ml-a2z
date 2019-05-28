#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:17:13 2019

@author: reyad
"""


# it's an example file
# i've created to teach and learn the new style
# to process data from 'data/Data.csv' file, you
# should consult 'data_preproc_new_style.py'

import numpy as np
import scipy
import pandas as pd
import matplotlib
import sklearn
import tensorflow as tf
import keras

data = {
        'Country' : ['France', 'Spain', 'Germany', 'Bangladesh',
                     'USA', 'France', 'Spain', 'Germany', 'Bangladesh', 'India'],
        'Age' : [np.nan, 43, 45, 67, 34, 23, 45, 90, 30, 90],
        'Salary' : [72000, 32000, 43567, 90000, 239000, np.nan, 90087, 40987, 45987, 35674],
        'Purchased': ['yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes']
}

df = pd.DataFrame(data)
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =============================================================================
# new ways to handle data
# =============================================================================
# there are several ways to purify data
# i can do it as the old ways, using Imputer, label decoder, OneHotEncoder etc
# but i'm going do it using the new way
# using ColumnTransformer
# for using ColumnTransformer class directly, i've to write much as shown below
# so i'm going to use make_column_transformer
# using make_column_transfer, i've to write less
# codes are shown below...
# also notice the use of pipeline
# for executing two commands one after another in common columns use pipeline
# using two transformers does not do the job, they just create more columns seperately
# you can test
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
# =============================================================================
# commented section of CoulumnTransformer use
# =============================================================================
#preprocess_X = ColumnTransformer(
#        transformers=[('one_hot_encoder', OneHotEncoder(), ['Country']),
#                      ('simple_imputer_standard_scaler', make_pipeline(
#                                         SimpleImputer(missing_values=np.nan, strategy='mean'),
#                                         StandardScaler()), ['Age', 'Salary']),
#                      ],
#        remainder='passthrough')
#preprocess_X.fit(X_train)
#X_train = preprocess_X.transform(X_train)

# =============================================================================
# using make_column_transformer
# =============================================================================
preprocess_X = make_column_transformer(
    (OneHotEncoder(), [0]),
    (make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy='mean'),
            StandardScaler()
        ),
        [1, 2]
    ),
    remainder = 'passthrough'
)
# =============================================================================
# Remember
# =============================================================================
# fit method only calculates the parameter for transformation
# and saves those parameter....so that any dataset like train and test both can be
# transformed
# fit_transform does the job of fit and
# then also transform the given dataset and return the result

preprocess_X.fit(X_train)
X_train = preprocess_X.transform(X_train)

# =============================================================================
# commented section using ColumnTransformer
# =============================================================================
# couldn't use LabelEncoder with ColumnTransformer
# error: expected two positional arguments...given 3
# solution: use OrdinalEncoder...does the same thing...but
# you've to use DataFrame object...may be it doesn't work on series..not sure though

#preprocess_y = ColumnTransformer(
#        transformers=[('label_encoder', OrdinalEncoder(), ['Purchased'])],
#        remainder='passthrough'
#)
#preprocess_y.fit(y_train)
#y_train = preprocess_y.fit_transform(y_train)


# =============================================================================
# using make_column_transformer
# =============================================================================
preprocess_y = make_column_transformer(
        (OrdinalEncoder(), [0]),
        remainder='passthrough'
)
preprocess_y.fit(y_train)
y_train = preprocess_y.transform(y_train)

