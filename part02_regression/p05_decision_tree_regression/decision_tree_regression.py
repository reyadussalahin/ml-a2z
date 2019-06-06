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
            'part02_regression/p05_decision_tree_regression/data/Position_Salaries.csv'
            
dataset = pd.read_csv(DATA_PATH)
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

X = np.array(X).reshape((len(X), 1))
y = np.array(y).reshape((len(y), 1))

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

# low resolution view
# it seems wrong i.e. overfitted to data
# but let's check with higher resolution first
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Decision Tree Regression(blue)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()


# higher resolution view
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = np.array(X_grid).reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression(blue)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# ok...in higher resolution it seems ok