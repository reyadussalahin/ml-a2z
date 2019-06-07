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
            'part08_deep_learning/p01_artificial_neural_networks/data/Churn_Modelling.csv'

dataset = pd.read_csv(DATA_PATH)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# preprocess data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
preprocess_X = make_column_transformer(
        (OneHotEncoder(), [1]),
        (OrdinalEncoder(), [2]),
        (StandardScaler(), [0, 3, 4, 5, 6, 7, 8, 9]),
        remainder='passthrough'
)
preprocess_X.fit(X_train)

X_train = preprocess_X.transform(X_train)
X_train = X_train[:, 1:]

X_test = preprocess_X.transform(X_test)
X_test = X_test[:, 1:]

# building model
from keras.models import Sequential
from keras.layers import Dense

# model declaration
classifier = Sequential()

# adding layers
# first hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))

# second hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))

# adding output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# compiling model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting model to train set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# predicting result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
