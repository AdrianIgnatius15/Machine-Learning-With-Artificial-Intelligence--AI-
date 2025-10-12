import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn import linear_model;
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error;

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv";
df = pd.read_csv(url);

# Select relevant features for the model
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']];
print("\nSelected features:");
print(cdf.head());
print("\nSample data:");
print(cdf.sample(9));

# Define features and target variable
X = cdf.ENGINESIZE.to_numpy();
y = cdf.FUELCONSUMPTION_COMB.to_numpy();

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);
type(X_train), np.shape(X_train), np.shape(X_train);

regressor = linear_model.LinearRegression();
# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train);
# The coefficients
print("\nCoefficients:", regressor.coef_[0]);
print("Intercept:", regressor.intercept_);

# Plot outputs
# plt.scatter(X_train, y_train, color="blue");
# plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r');
# plt.xlabel("Engine Size");
# plt.ylabel("CO2 Emissions");
# plt.show();

y_pred = regressor.predict(X_test.reshape(-1, 1));
# Explained variance score: 1 is perfect prediction
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred));
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred));
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)));
print("R2-score: %.2f" % r2_score(y_test, y_pred));

## PRACTICE
# Plot the regression line over the test data
plt.scatter(X_test, y_test, color="pink");
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-g');
plt.xlabel("Engine Size");
plt.ylabel("Fuel Consumption");
plt.show();