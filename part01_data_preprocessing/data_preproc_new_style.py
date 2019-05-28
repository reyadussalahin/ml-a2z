import numpy as np
import pandas as pd

DATA_PATH = '~/Codes/mlproject/keras_tf_conda/project/ml_a2z/part01_data_preprocessing/data/Data.csv'

dataset = pd.read_csv(DATA_PATH)
X = dataset.iloc[:, :-1] # independent column(s)
y = dataset.iloc[:, -1:] # dependent column(s)


# i'm going preprocess data the new way

# =============================================================================
# Column 'Country'
# =============================================================================
# as you know, first column is 'Country'
# which is a Categorical variable
# so, we have to label_encode and one_hot_encode it(according to old ways)

# But now scikit-learn can does this job only using one_hot_encoder
# so no need to use label_encoder
# one_hot_encoder are now able to this...because
# it now works by selecting unique values...while previously it works
# on basis of 0->max(values)
# so, hope you get it properly

# =============================================================================
# Column 'Age' and 'Salary'
# =============================================================================
# for age and salary, we've to fill the empty position
# and we also need to do feature scaling as salary_range >>> age_range
# previously what we've done is, first we used Imputer, then
# split the data into train and test set using train_test_split
# then we used StandardScaler.fit_tranform() on X_train and then use that
# StandardScaler object to transform X_test

# But what we are going to do now is quite different
# As, you may assume
# Everything transform on test_data should be based on train_data
# so, what we have to do is split the data first into train and test data
# then we'll use ColumnTransformer or make_column_transfer(I'm going to use it)
# to apply transformers as what column(s) need's what type of transformations
# [n.b.]: for two consecutive transformations, like 'age' and 'salary' columns
# which need at first 'fill nan(empty) value' and then 'feature scaling', i'm
# going to use make_pipeline() function
# the reason, i'm using pipeline is, applying two transformers consecutively in
# common columns doesn't change these common column twice, rather it creates
# two different instances of the column columns
# but pipeline does the job properly
# pipeline has been for applying consecutive operations on columns or data
# which need common operations or transformations


# =============================================================================
# Column 'Purchased'
# =============================================================================
# we've used label_encoder previously(old way)

# But now we're going to use OrdinalEncoder()[please use dataframe object with
# this encoder]
# cause, label_encoder doesn't work with ColumnTransformer
# that's why we are going to use ordinal_encoder which does the same job
# and also works with ColumnTransformer

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

preprocess_X = make_column_transformer(
        (OneHotEncoder(), [0]), # applying one_hot_encoder to 'Country' column
        (make_pipeline(
                SimpleImputer(missing_values=np.nan, strategy='mean'),
                StandardScaler()),
            [1, 2]
        ), # applying SimpleImputer first, then StandardScaler on
           # 'Age' & 'Salary'
        remainder='passthrough' # it's import, because by default
        # remainder='drop', so if we don't do this, the column those are
        # unaffected would be dropped, we don't want that behaviour
        # we want all the columns, whether it is changed or not.
)
preprocess_X.fit(X_train) # fitting the model using X_train
# fit determines the parameters to transform column(i.e. feature) values
# so, preprocess_X will also be used to transform X_test values
X_train = preprocess_X.transform(X_train) # changing X_train values

preprocess_y = make_column_transformer(
        (OrdinalEncoder(), [0]), # changing 'Purchased' column to
        # numerical values
        remainder='passthrough'
)
preprocess_y.fit(y_train) # fitting the model using y_train
# so, we will use to transform both y_train and y_test(later)
y_train = preprocess_y.transform(y_train)


# processing test data before feeding to model
X_test = preprocess_X.transform(X_test)
y_test = preprocess_y.transform(y_test)

# now you can feed the model X_test
# ...
# ... do what you need to do
# ...

# and so, we're done
# =============================================================================
#                                  THE END
# =============================================================================
