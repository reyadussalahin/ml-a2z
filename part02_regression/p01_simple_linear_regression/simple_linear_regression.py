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


# regression section begins
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = PROJECT_ROOT + \
            'part02_regression/p01_simple_linear_regression/data/Salary_Data.csv'

dataset = pd.read_csv(DATA_PATH)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# importing linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

y_pred_train = regressor.predict(X_train)

# plotting point to see evaluate results
plt.plot(X_train, y_pred_train, color='blue')
plt.scatter(X_train, y_train, color='red')
plt.scatter(X_test, y_test, color='green')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experinece')
plt.ylabel('Salary')
plt.show()
