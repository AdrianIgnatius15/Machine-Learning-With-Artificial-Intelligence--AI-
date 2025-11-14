"""
=================================
K-Nearest Neighbours (KNN)
=================================
This module implements the K-Nearest Neighbours algorithm for supervised learning tasks.
After completing this lab you will be able to:
Use K-Nearest neighbors to classify data
Apply KNN classifier on a real world data set
"""

import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
import seaborn as sns;
from sklearn.preprocessing import StandardScaler;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.metrics import accuracy_score;

"""
======================
About the Data Set
======================
Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. 
If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. 
It is a classification problem. 
That is, given the dataset, with predefined labels, we need to build a model to be used to predict class of a new or unknown case.
The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.
The target field, called custcat, has four possible service categories that correspond to the four customer groups, as follows:

1. Basic Service
2. E-Service
3. Plus Service
4. Total Service
Our objective is to build a classifier to predict the service category for unknown cases. We will use a specific type of classification called K-nearest neighbors.
"""

# Load the data set
url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv";
dataFrame: pd.DataFrame = pd.read_csv(url);
print("Data Initial Info: \n", dataFrame.head());

# Data Visualization
custcat_counts: pd.Series = dataFrame['custcat'].value_counts();
print("\nCustomer Category Counts: \n", custcat_counts);
correlation_matrix: pd.DataFrame = dataFrame.corr();
plt.figure(figsize=(10, 8));
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5);
##### plt.show();

## Separate the input features and target feature
X_axis: pd.DataFrame = dataFrame.drop('custcat', axis=1);
y_axis: pd.Series = dataFrame['custcat'];

# Normalize the data
### Data normalisation is important for KNN algorithm as it is based on distance metrics.
### This is to ensure that all features contribute equally to the distance calculation.
X_normalized_axis: np.ndarray = StandardScaler().fit_transform(X_axis);
## Train-Test Split
### Now, you should separate the training and the testing data.
### You can retain 20% of the data for testing purposes and use the rest for training. 
### Assigning a random state ensures reproducibility of the results across multiple executions.
X_train, X_test, y_train, y_test = train_test_split(X_normalized_axis, y_axis, test_size=0.2, random_state=4);


# KNN Classifier
## Training the Model
### Initially, you may start by using small value as the value as 'k' such as 3. Let's try `k=4`.
k: int = 30;
knn_classifier: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=k);
knn_model: KNeighborsClassifier = knn_classifier.fit(X_train, y_train);

## Predicting
### Once the model is trained, we can now use this model to generate predictions for the test set.
y_axis_predicted = knn_model.predict(X_test);

## Accuracy evaluation
## In multilabel classification, accuracy classification score is a function that computes subset accuracy.
## This function is equal to the jaccard_score function.
## Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
print("Test set accuracy score", accuracy_score(y_test, y_axis_predicted));

# Choosing the correct value of `k`
"""
K in KNN, is the number of nearest neighbors to examine. However, the choice of the value of 'k' clearly affects the model. Therefore, the appropriate choice of the value of the variable k becomes an important task. The general way of doing this is to train the model on a set of different values of k and noting the performance of the trained model on the testing set. The model with the best value of accuracy_score is the one with the ideal value of the parameter k.
Check the performance of the model for 10 values of k, ranging from 1-9. You can evaluate the accuracy along with the standard deviation of the accuracy as well to get a holistic picture of the model performance.
"""
# ks: int = 10;
# acc = np.zeros((ks));
# std_acc = np.zeros((ks));
# for n in range(1, ks + 1):
#     ### Train the model and predict as `n` increases
#     knn_model_n: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train);
#     y_predicted_axis = knn_model_n.predict(X_test);
#     acc[n - 1] = accuracy_score(y_test, y_predicted_axis);
#     std_acc[n - 1] = np.std(y_predicted_axis == y_test) / np.sqrt(y_predicted_axis.shape[0]);

# Plot the model accuracy for a different number of neighbours.
## Now, you can plot the model accuracy and the standard deviation to identify the model with the most suited value of `k`
# plt.plot(range(1,ks + 1),acc,'g');
# plt.fill_between(range(1, ks + 1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10);
# plt.legend(('Accuracy value', 'Standard Deviation'));
# plt.ylabel('Model Accuracy');
# plt.xlabel('Number of Neighbors (K)');
# plt.tight_layout();
# plt.show();
# print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1);