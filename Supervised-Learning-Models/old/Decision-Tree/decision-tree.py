# DECISION TREE
## Objective:
"""
### Develop a classification model using Decision Tree Algorithm
### Apply Decision Tree classification on a real-world dataset.
"""

# Introduction
## This lab explores decision tree classification, a powerful machine learning technique for making data-driven decisions. 
## You will learn to build, visualize, and evaluate decision trees using a real-world dataset. 
## The dataset used in this lab is that of Drug prediction based on the health parameters of a patient.

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import warnings;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;
from sklearn.tree import DecisionTreeClassifier, plot_tree;
from sklearn import metrics;

warnings.filterwarnings("ignore");

# About the dataset
## Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug C, Drug X and Drug Y.
## Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The features of this dataset are the Age, Sex, Blood Pressure, and Cholesterol of the patients, and the target is the drug that each patient responded to.
## It is a sample of a multiclass classifier, and you can use the training part of the dataset to build a decision tree, and then use it to predict the class of an unknown patient or to prescribe a drug to a new patient.

# Download the dataset

path: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv";
my_data: pd.DataFrame = pd.read_csv(path);

# Data Analysis and Preprocessing
# print(my_data.info());

## We found out that 4 columns have object datatype. We need to convert them into numerical values using LabelEncoder.
## Otherwise, machine learning cannot work with categorical data directly.
le: LabelEncoder = LabelEncoder();
my_data["Sex"] = le.fit_transform(my_data["Sex"]);
my_data["BP"] = le.fit_transform(my_data["BP"]);
my_data["Cholesterol"] = le.fit_transform(my_data["Cholesterol"]);

## So evaluate the correlation between the target variable (Drug) and the features (Sex, BP and Cholesterol),
## we convienently map the different drug types to a numerical value.
my_data["Drug"] = my_data["Drug"].map({"drugA": 0, "drugB": 1, "drugC": 2, "drugX": 3, "drugY": 4});

## We can also understand the distribution of the dataset by plotting the count of the records with each drug recommendation.
# category_counts: pd.Series = my_data["Drug"].value_counts();
# plt.bar(category_counts.index, category_counts.values.astype(float), color='blue');
# plt.xlabel('Drug');
# plt.ylabel('Count');
# plt.title('Category Distribution');
# plt.xticks(rotation=45);
# plt.show();

# Modeling
## We will split the dataset into training and testing sets.
y = my_data["Drug"];
X = my_data.drop(columns=["Drug"]);
## Split the dataset into training and testing sets with 30% for testing and random state of 32
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32);
## We can then create a Decision Tree classifier object as "DrugTree" and fit the training data to the model.
drugTree: DecisionTreeClassifier = DecisionTreeClassifier(criterion="entropy", max_depth=4);
drugTree.fit(X_trainset, y_trainset);

# Evaluation
## We can make predictions on the testing dataset and store it into a variable called "tree_predictions".
tree_predictions: np.ndarray = drugTree.predict(X_testset);
## We can now check the accuracy of our model by comparing the actual values of y_testset with the predicted values stored in tree_predictions.
print("Decision Tree's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions));

# Visualization
## We can visualize the decision tree
plot_tree(drugTree);
plt.show();

## PRACTICE QUESTIONS
"""
1. Along similar lines, identify the decision criteria for all other classes.
2. If the max depth of the tree is reduced to 3, how would the performance of the model be affected?
"""
## 1. Ans:

