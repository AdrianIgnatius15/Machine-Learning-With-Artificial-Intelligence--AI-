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