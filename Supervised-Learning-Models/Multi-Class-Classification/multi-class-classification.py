# Multi-class Classification
################################################
# This code file contains and implements logistic regression for multi-class classification problems using the One-vs-All (OvA) strategy & One-vs-One (OvO) strategy.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
import warnings;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import OneHotEncoder, StandardScaler;
from sklearn.linear_model import LogisticRegression;
from sklearn.multiclass import OneVsOneClassifier;
from sklearn.metrics import accuracy_score;

warnings.filterwarnings("ignore");

"""
# About the Dataset
The data used will be "Obesity Risk Prediction" dataset from the UCI Machine Learning Repository.
The data set has 17 attributes in total along with 2,111 samples.
"""
# Load the dataset
url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv";
data = pd.read_csv(url);
# Display the first 5 rows of the dataset
print(data.head());

# EXEPLORATORY DATA ANALYSIS (EXERCISE)
## Visualise the distrbution of the target variable to understand the class balance.
# sns.countplot(x='NObeyesdad', data=data);
# plt.title('Distribution of Obesity Levels');
# plt.ylabel('Count');
# plt.show();

## Check for null values and display summary statistics of the dataset.
print(data.isnull().sum());
print(data.info());
print(data.describe());

# Data Preprocessing
## Feature scaling: Scale the numerical features to standardize their range for better model performance.
## Standardization of data is important to better define the decision boundaries between classes by making sure that the feature variations are in similar scales. The data is now ready to be used for training and testing.
continous_columns = data.select_dtypes(include=['float64']).columns.tolist();
scaler: StandardScaler = StandardScaler();
scaled_features = scaler.fit_transform(data[continous_columns]);

## Converting to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continous_columns));

## Combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continous_columns), scaled_df], axis=1);

# One-Hot Encoding
## Convert categorical variables into numerical format using one-hot encoding.
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist();
categorical_columns.remove('NObeyesdad'); # Exclude target variable

## Applying One-Hot Encoding
encoder: OneHotEncoder = OneHotEncoder(sparse_output=False, drop='first');
encoded_features = encoder.fit_transform(scaled_data[categorical_columns]);

## Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns));

## Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1);

# Encoding Target Variable
## Convert target variable into numerical format
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes;
# Display the first 5 rows of the preprocessed dataset
print(prepped_data.head());

# Separate the input and target data
X = prepped_data.drop(columns=['NObeyesdad']).values;
y = prepped_data['NObeyesdad'];

"""
# Train-Test Split
Split the dataset into training and testing sets to evaluate model performance on unseen data.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y);

## One-vs-All (OvA) Logistic Regression
### Create an instance of the Logistic Regression model using OvA strategy
model_ova: LogisticRegression = LogisticRegression(multi_class='ovr', max_iter=1000).fit(X_train, y_train);

## Predictions on data set
y_pred_ova = model_ova.predict(X_test);

## Evaluation metrics for OvA model
print("One-vs-All (OvA) Logistic Regression Model Accuracy: ", np.round(accuracy_score(y_test, y_pred_ova), 2) + "%");

## One-vs-One (OvO) Logistic Regression
### Create an instance of the Logistic Regression model using OvO strategy
model_ovo: OneVsOneClassifier = OneVsOneClassifier(LogisticRegression(max_iter=1000)).fit(X_train, y_train);

## Predictions on data set
y_pred_ovo = model_ovo.predict(X_test);

## Evaluation metrics for OvO model
print("One-vs-One (OvO) Logistic Regression Model Accuracy: ", np.round(accuracy_score(y_test, y_pred_ovo), 2) + "%");

# Conclusion
"""
In this code, we implemented logistic regression for multi-class classification using both One-vs-All (OvA) and One-vs-One (OvO) strategies. 
We preprocessed the data by scaling numerical features and applying one-hot encoding to categorical variables. 
Finally, we evaluated the performance of both models using accuracy as the metric.
"""

### EXERCISE
####Q1: Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3) and observe the impact on model performance.
#### X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y);

####Q2: Plot a bar chart of feature importance using the coefficients from the One vs All logistic regression model. Also try for the One vs One model.
#### coefficients_ova = pd.Series(model_ova.coef_[0], index=prepped_data.columns[:-1]);
#### coefficients_ova.sort_values().plot(kind='barh', title='Feature Coefficients (OvA)', xlabel='Coefficient Value', ylabel='Feature');
#### plt.title("Feature Coefficients in One-vs-All Logistic Regression Model");
#### plt.show();

####Q3 Write a function `obesity_risk_pipeline` to automate the entire pipeline:
#### 1. Loading and preprocessing the data
#### 2. Training the model
#### 3. Evaluating the model
#### The function should accept the file path or URL and test size as parameters
