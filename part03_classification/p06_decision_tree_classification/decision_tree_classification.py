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
            'part03_classification/p06_decision_tree_classification/data/Social_Network_Ads.csv'

dataset = pd.read_csv(DATA_PATH)
X = dataset.iloc[:, [2, 3]]
y = dataset.iloc[:, 4]

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)


# preprocess data
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
preprocess_X = make_column_transformer(
        (StandardScaler(), [0, 1]),
        remainder='passthrough'        
)
preprocess_X.fit(X_train)
X_train = preprocess_X.transform(X_train)
X_test = preprocess_X.transform(X_test)


# building model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# predicting result
y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# visualising training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
        np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01)
)
plt.contourf(
        X1, X2,
        classifier.predict(
                np.array([X1.ravel(), X2.ravel()]).T
        ).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c = ListedColormap(('red', 'green'))(i),
            label=j
    )
plt.title('Naive Bayes(Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# visualising test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
        np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01)
)
plt.contourf(
        X1, X2,
        classifier.predict(
                np.array([X1.ravel(), X2.ravel()]).T
        ).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c = ListedColormap(('red', 'green'))(i),
            label=j
    )
plt.title('Naive Bayes(Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
