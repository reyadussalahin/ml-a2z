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
            'part02_regression/p06_random_forest_regression/data/Position_Salaries.csv'

dataset = pd.read_csv(DATA_PATH)
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

X = X.reshape((len(X), 1))
y = y.reshape((len(y), 1))

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])


# high resolution view
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision tree regression(blue)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# as n_estimators value increases result gets more perfect