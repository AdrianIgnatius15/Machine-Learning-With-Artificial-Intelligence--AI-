import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression;
from sklearn.preprocessing import StandardScaler;
from sklearn.metrics import log_loss;

"""
## Logistic Regression

This code demonstrates how logistic regression is used.

### Case Study: Predicting Customer Churn
Telco Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base. 
Each case corresponds to a separate customer and it records various demographic and service usage information. 
Before you can work with the data, you must use the URL to get the ChurnData.csv.

### Data Set Description
We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company.

This data set provides you information about customer preferences, services opted, personal details, etc. which helps you predict customer churn.
"""

# Load the dataset
url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv";
churn_df: pd.DataFrame = pd.read_csv(url);

# Data Preprocessing
## For this lab, we can use a subset of the fields available to develop out model. Let us assume that the fields we use are 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip' and of course 'churn'.
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']];
churn_df['churn'] = churn_df['churn'].astype('int');

## Define input fields X and target field 'y'. In this case, 'churn' is the target field which is 'y' axis and the array of columns we have previously.
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]);
X[0:5]; # Display first 5 rows of X
y = np.asarray(churn_df['churn']);
y[0:5]; # Display first 5 rows of y

# Standardize the dataset
## It is a good practice to standardize the dataset before applying any machine learning algorithm. Standardization of a dataset involves rescaling the features so that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one.
X_norm = StandardScaler().fit(X).transform(X);
X_norm[0:5]; # Display first 5 rows of standardized X

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4);

# Logistic Regression Model
## Create an instance of the Logistic Regression model
LR: LogisticRegression = LogisticRegression().fit(X_train, y_train);
## Make predictions on the test set
yhat = LR.predict(X_test);
yhat[:10]; # Display first 10 predictions
## Predict probabilities
yhat_prob = LR.predict_proba(X_test);
yhat_prob[:10]; # Display first 10 predicted probabilities
## 1 class prediction. The purpose here is to predict the 1 class more acccurately, you can also examine what role each input feature has to play in the prediction of the 1 class.
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1]);
coefficients.sort_values().plot(kind='barh', title='Feature Coefficients', xlabel='Coefficient Value', ylabel='Feature');
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

# Model Evaluation
## Compute Log Loss
logloss = log_loss(y_test, yhat_prob);
print(f"Log Loss: {logloss}");