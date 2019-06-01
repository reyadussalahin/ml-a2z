# it's the part same in every script
# please change your project_root dir as you need
# this project_root may not be same as yours...running the code won't work
# this whole process i've gone through to provide Data file path
# you can/should follow your own approach
import os

if os.name == 'posix':
    PROJECT_ROOT = '~/Codes/mlproject/keras_tf_conda/project/ml_a2z/'
else:
    PROJECT_ROOT = 'G:/REYAD/CODES/mlproject/keras_tf/project/ml_a2z/'


# multiple_linear_regression started
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = PROJECT_ROOT + \
            'part02_regression/p02_multiple_linear_regression/data/50_Startups.csv'

dataframe = pd.read_csv(DATA_PATH)
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
preprocessor_X = make_column_transformer(
        (OneHotEncoder(), ['State']), # applying one_hot_encoder to 'State' column
        remainder='passthrough'
)
preprocessor_X.fit(X_train)
X_train = preprocessor_X.transform(X_train)

X_train = X_train[:, 1:]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

X_test = preprocessor_X.transform(X_test)
X_test = X_test[:, 1:]
y_pred = regressor.predict(X_test)


# import statsmodels.formula.api as sm # can't import this module
# Error shown: "ModuleNotFoundError: No module named 'statsmodels'"
# have to look into it
