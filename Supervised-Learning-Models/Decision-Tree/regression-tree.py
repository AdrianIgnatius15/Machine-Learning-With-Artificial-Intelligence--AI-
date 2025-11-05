from __future__ import print_function;
# REGRESSION TREES
"""
In this exercise session you will use a real dataset to train a regression tree model. 
The dataset includes information about taxi tip and was collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorized under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). 
You will use the trained model to predict the amount of tip paid.
"""

# Introduction
"""
The dataset used in this exercise session is a subset of the publicly available TLC Dataset (all rights reserved by Taxi & Limousine Commission (TLC), City of New York). 
The prediction of the tip amount can be modeled as a regression problem. 
To train the model you can use part of the input dataset and the remaining data can be used to assess the quality of the trained model.
"""

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import normalize;
from sklearn.metrics import mean_squared_error;

import warnings;
warnings.filterwarnings('ignore');

# Dataset Analysis
## Load the dataset from source.
url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv";
raw_data: pd.DataFrame = pd.read_csv(url);
print(raw_data.info());

## Since this data is about predicting the tip amount, let's understand the dataset a little better with correlating between the target variable against the input variables.
correlations_values: pd.Series = raw_data.corr()['tip_amount'].drop('tip_amount');
correlations_values.plot(kind='barh', figsize=(10, 6));


# Data Preprocessing
## Let's prepare the data for training by applying normalisation to the input features.
## Extract the labels from the dataframe.
y = raw_data[['tip_amount']].values.astype(np.float32);
## drop the target variable from the feature matrix.
formatted_data = raw_data.drop(['tip_amount'], axis=1);
## Get the feature matrix (the input variables).
X = formatted_data.values;
## Normalize the feature matrix.
X = normalize(X, axis=1, norm="l1", copy=False);

# Dataset Model Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);

"""
Build a Decision Tree Regressor model with Scikit-Learn
Regression Trees are implemented using DecisionTreeRegressor.

The important parameters of the model are:

criterion: The function used to measure error, we use 'squared_error'.

max_depth - The maximum depth the tree is allowed to take; we use 8.
"""
from sklearn.tree import DecisionTreeRegressor;
dt_regressor: DecisionTreeRegressor = DecisionTreeRegressor(criterion='squared_error', max_depth=12, random_state=35);
dt_regressor.fit(X_train, y_train);

# Model Evaluation
"""
To evaluate our dataset we will use the score method of the DecisionTreeRegressor object providing our testing data, this number is the 
value which indicates the coefficient of determination. We will also evaluate the Mean Squared Error 
of the regression output with respect to the test set target values. High 
and low values are expected from a good regression model.
"""
## Run inference using sklearn model.
y_pred = dt_regressor.predict(X_test);
## Evaluate mean squared error on the test dataset.
mse_score = mean_squared_error(y_test, y_pred);
r2_score = dt_regressor.score(X_test, y_test);
print(f"Mean Squared Error (MSE) on test dataset: {mse_score:.4f}");
print(f"R2 Score on test dataset: {r2_score:.4f}");

## PRACTICE
'''
Identify the top 3 features that have the highest correlation with the tip amount in the dataset.
'''
top_3_features = correlations_values.abs().sort_values(ascending=False).head(3)
print("Top 3 features with highest correlation to tip amount:", top_3_features.index.tolist());

'''
Identify 4 features that have very low or no correlation with the tip amount and remove them from the dataset before training the model.
'''
top_4_low_corr_features = correlations_values.abs().sort_values(ascending=True).head(4)
print("4 features with lowest correlation to tip amount:", top_4_low_corr_features.index.tolist());
formatted_data_reduced = raw_data.drop(['tip_amount'] + top_4_low_corr_features.index.tolist(), axis=1);
X_reduced = formatted_data_reduced.values;
X_reduced = normalize(X_reduced, axis=1, norm="l1", copy=False);
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y, test_size=0.3, random_state=42);
dt_regressor_reduced = DecisionTreeRegressor(criterion='squared_error', max_depth=12, random_state=35);
dt_regressor_reduced.fit(X_train_reduced, y_train_reduced);
y_pred_reduced = dt_regressor_reduced.predict(X_test_reduced);
mse_score_reduced = mean_squared_error(y_test_reduced, y_pred_reduced);
r2_score_reduced = dt_regressor_reduced.score(X_test_reduced, y_test_reduced);
print(f"Mean Squared Error (MSE) on reduced test dataset: {mse_score_reduced:.4f}");
print(f"R2 Score on reduced test dataset: {r2_score_reduced:.4f}");

'''
Check the effect of decreasing the max_depth parameter of the DecisionTreeRegressor to 8 and 4 on the model's performance.
'''
for depth in [8, 4]:
    dt_regressor_depth = DecisionTreeRegressor(criterion='squared_error', max_depth=depth, random_state=35);
    dt_regressor_depth.fit(X_train, y_train);
    y_pred_depth = dt_regressor_depth.predict(X_test);
    mse_score_depth = mean_squared_error(y_test, y_pred_depth);
    r2_score_depth = dt_regressor_depth.score(X_test, y_test);
    print(f"Max Depth: {depth} -> Mean Squared Error (MSE): {mse_score_depth:.4f}, R2 Score: {r2_score_depth:.4f}");