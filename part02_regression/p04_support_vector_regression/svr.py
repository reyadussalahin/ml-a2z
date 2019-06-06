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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = PROJECT_ROOT + \
            'part02_regression/p04_support_vector_regression/data/Position_Salaries.csv'

dataset = pd.read_csv(DATA_PATH)
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

X = X.reshape((len(X), 1))
y = y.reshape((len(y), 1))

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
preprocess_X = make_column_transformer(
        (StandardScaler(), [0]),
        remainder='passthrough'
)
preprocess_X.fit(X)
X = preprocess_X.transform(X)

preprocess_y = make_column_transformer(
        (StandardScaler(), [0]),
        remainder='passthrough'
)
preprocess_y.fit(y)
y = preprocess_y.transform(y)

# building model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)


# predicting value
sc_y = preprocess_y.named_transformers_.get('standardscaler')
y_pred = sc_y.inverse_transform(regressor.predict(preprocess_X.transform([[6.5]])))


# predicting salary value using svr model
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Support Vector Regression(blue)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()
