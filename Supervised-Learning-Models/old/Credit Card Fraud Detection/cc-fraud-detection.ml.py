from __future__ import print_function;
"""
Credit Card Fraud Detection with Decision Trees and SVM
"""

"""
Consolidate your machine learning (ML) modeling skills by using two popular classification models to identify fraudulent credit card transactions. 
These models are: Decision Tree and Support Vector Machine. 
You will use a real dataset of credit card transactions to train each of these models. 
You will then use the trained model to assess if a credit card transaction is fraudulent or not.

Imagine that you work for a financial institution and part of your job is to build a model that predicts if a credit card transaction is fraudulent or not. You can model the problem as a binary classification problem. A transaction belongs to the positive class (1) if it is a fraud, otherwise it belongs to the negative class (0).
You have access to transactions that occured over a certain period of time. The majority of the transactions are normally legitimate and only a small fraction are non-legitimate. Thus, typically you have access to a dataset that is highly unbalanced. This is also the case of the current dataset: only 492 transactions out of 284,807 are fraudulent (the positive class - the frauds - accounts for 0.172% of all transactions).
This is a Kaggle dataset. You can find this "Credit Card Fraud Detection" dataset from the following link: Credit Card Fraud Detection.
To train the model, you can use part of the input dataset, while the remaining data can be utilized to assess the quality of the trained model. First, let's import the necessary libraries and download the dataset.
"""

import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import normalize, StandardScaler;
from sklearn.utils.class_weight import compute_sample_weight;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.metrics import roc_auc_score;
from sklearn.svm import LinearSVC;

import warnings;
warnings.filterwarnings('ignore');

# Load the dataset
url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv";
raw_data: pd.DataFrame = pd.read_csv(url);
print("Dataset info:", raw_data.info());


# Data Analysis
"""
Each row in the dataset represents a credit card transaction. 
As shown above, each row has 31 variables. One variable (the last variable in the table above) is called Class and represents the target variable. 
Your objective will be to train a model that uses the other variables to predict the value of the Class variable. Let's first retrieve basic statistics about the target variable.
Note: For confidentiality reasons, the original names of most features are anonymized V1, V2 .. V28. The values of these features are the result of a PCA transformation and are numerical. 
The feature 'Class' is the target variable and it takes two values: 1 in case of fraud and 0 otherwise. For more information about the dataset please visit this webpage: https://www.kaggle.com/mlg-ulb/creditcardfraud.
"""
## Get the set of distinct class labels
labels = raw_data["Class"].unique().tolist();
## Get the count of each class label
sizes = raw_data["Class"].value_counts().to_numpy();
# plot the class value counts
fig, ax = plt.subplots();
ax.pie(sizes, labels=labels, autopct='%1.3f%%');
ax.set_title('Target Variable Value Counts');
# plt.show();

"""
As shown above, the Class variable has two values: 0 (the credit card transaction is legitimate) and 1 (the credit card transaction is fraudulent). 
Thus, you need to model a binary classification problem. Moreover, the dataset is highly unbalanced, the target variable classes are not represented equally. This case requires special attention when training or when evaluating the quality of a model. 
One way of handing this case at train time is to bias the model to pay more attention to the samples in the minority class. The models under the current study will be configured to take into account the class weights of the samples at train/fit time.

It is also prudent to understand which features affect the model in what way. We can visualize the effect of the different features on the model using the code below.
"""
## Understand the features affects
correlation_values = raw_data.corr()['Class'].drop('Class');
correlation_values.plot.bar(figsize=(10, 6));

# Data Preprocessing
"""
You will now prepare the data for training. You will apply standard scaling to the input features and normalize them using 
 norm for the training models to converge quickly. As seen in the data snapshot, there is a parameter called Time which we will not be considering for modeling. 
Hence, features 2 to 30 will be used as input features and feature 31, i.e. Class will be used as the target variable.
"""
## Standardise the input features by removing the mean and scaling to unit variance
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30]);
data_matrix = raw_data.values;
## X: feature matrix (for this analysis, we exclude the Time column)
X = data_matrix[:, 1:30];
## y: target variable which is the Class column
y = data_matrix[:, 30];
## data normalisation
X = normalize(X, norm="l1");


# Dataset Training and Testing Split
"""
Now that the dataset is ready for building the classification models, 
you need to first divide the pre-processed dataset into a subset to be used for training the model (the train set)
and a subset to be used for evaluating the quality of the model (the test set).
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42);

# Build a Decision Tree Classifier model with Scikit-learn
## Compute the sample weights to be used as input to the train routine so that it takes into account the class imbalance present in this dataset.
w_train = compute_sample_weight(class_weight="balanced", y=y_train);
## Create the Decision Tree Classifier model
dt_classifier: DecisionTreeClassifier = DecisionTreeClassifier(max_depth=5, random_state=35);
dt_classifier.fit(X_train, y_train, sample_weight=w_train);

# Build a Support Vector Machine Classifier model with Scikit-learn
svm = LinearSVC(class_weight="balanced", random_state=31, loss="hinge", fit_intercept=False);
svm.fit(X_train, y_train);

# Evaluate the Decision Tree Classifier model
y_pred_dt = dt_classifier.predict(X_test);
"""
Using these probabilities, we can evaluate the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) score as a metric of model performance. 
The AUC-ROC score evaluates your model's ability to distinguish positive and negative classes considering all possible probability thresholds. 
The higher its value, the better the model is considered for separating the two classes of values.
"""
roc_auc_dt = roc_auc_score(y_test, y_pred_dt);
print("Decision Tree Classifier ROC-AUC score: {:.4f}".format(roc_auc_dt));

# Evaluate the Support Vector Machine Classifier model
y_pred_svm = svm.decision_function(X_test);
roc_auc_svm = roc_auc_score(y_test, y_pred_svm);
print("Support Vector Machine Classifier ROC-AUC score: {:.4f}".format(roc_auc_svm));

## PRACTICE EXERCISE
"""
Q1: Currently, we have used all 30 features of the dataset for training the models. Use the corr() function to find the top 6 features of the dataset to train the models on.
"""
correlation_values = raw_data.corr()['Class'].drop('Class');
top_6_features = correlation_values.abs().sort_values(ascending=False).head(6).index.tolist();
print("Top 6 features correlated with Class:", top_6_features);

"""
Q2: Using only these 6 features, modify the input variable for training
"""
X_top_6_features = raw_data[top_6_features].values;
X_top_6_features = normalize(X_top_6_features, norm="l1");
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X_top_6_features, y, test_size=0.3, random_state=42);

"""
Q3: Execute the Decision Tree model for this modified input variable. How does the value of the ROC-AUC score change?
"""
w_train_for_top_6_features = compute_sample_weight(class_weight="balanced", y=y_train_6);
dt_classifier_6 = DecisionTreeClassifier(max_depth=5, random_state=35);
dt_classifier_6.fit(X_train_6, y_train_6, sample_weight=w_train_for_top_6_features);
y_pred_dt_6 = dt_classifier_6.predict(X_test_6);
roc_auc_dt_6 = roc_auc_score(y_test_6, y_pred_dt_6);
print("Decision Tree Classifier ROC-AUC score with top 6 features: {:.4f}".format(roc_auc_dt_6));

"""
Q4: Execute the Support Vector Machine model for this modified input variable. How does the value of the ROC-AUC score change?
"""
svm = LinearSVC(class_weight="balanced", random_state=31, loss="hinge", fit_intercept=False);
svm.fit(X_train_6, y_train_6);
y_pred_svm_6 = svm.decision_function(X_test_6);
roc_auc_svm_6 = roc_auc_score(y_test_6, y_pred_svm_6);
print("Support Vector Machine Classifier ROC-AUC score with top 6 features: {:.4f}".format(roc_auc_svm_6));