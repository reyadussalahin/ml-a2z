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
            'part02_regression/p03_polynomial_regression/data/Position_Salaries.csv'

dataset = pd.read_csv(DATA_PATH)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=4)
X_poly = pf.fit_transform(X)

lreg2 = LinearRegression()
lreg2.fit(X_poly, y)


# linear regression prediction graph
plt.scatter(X, y, color='red')
plt.plot(X, lreg.predict(X), color='blue')
plt.title('Linear Regression(blue)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# polynomial regression predicton graph
plt.scatter(X, y, color='red')
plt.plot(X, lreg2.predict(pf.fit_transform(X)), color='blue')
plt.title('Polynomial Regression(blue)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# polynomial regression predicton graph in higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lreg2.predict(pf.fit_transform(X_grid)), color='blue')
plt.title('Polynomial Regression(blue)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

# linear regression prediction(see how inaccurate)
lreg.predict([[6.5]])
# result: array([330378.78787879])

# polynomial regression predicton(compare it with real value and linear prediction value)
lreg2.predict(pf.fit_transform([[6.5]]))
# result: array([158862.45265153])
