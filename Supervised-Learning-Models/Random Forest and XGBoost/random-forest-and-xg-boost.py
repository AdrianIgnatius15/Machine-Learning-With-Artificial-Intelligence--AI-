import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.datasets import fetch_california_housing;
from sklearn.model_selection import train_test_split;
from sklearn.ensemble import RandomForestRegressor;
from xgboost import XGBRegressor;
from sklearn.metrics import mean_squared_error, r2_score;
import time;

# Load the dataset of 1990 California Housing
## It is the cleanest data from California Housing.
data = fetch_california_housing()
X_axis, y_axis = data.data, data.target

# Split data for test and train
X_train, X_test, y_train, y_test = train_test_split(X_axis, y_axis, test_size=0.2, random_state=42);

# EXERCISE 1: How many observations and features does the dataset have?
N_observation, N_features = X_axis.shape;
print('Number of Observations: ' + str(N_observation));
print('Number of Features: ' + str(N_features));

# Initialise models
n_estimators: int = 100;
rf: RandomForestRegressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42);
xgb: XGBRegressor = XGBRegressor(n_estimators=n_estimators, random_state=42);
## Fit models
### Measure training the time for Random Forest
start_time_rf: float = time.time();
rf.fit(X_train, y_train);
end_time_rf: float = time.time();
total_random_forest_regressor_time: float = end_time_rf - start_time_rf;
print("Training time for Random Forrest Regressor:", total_random_forest_regressor_time);
### Measure training time for XGBoost
start_time_xgb: float = time.time();
xgb.fit(X_train, y_train);
end_time_xgb: float = time.time();
total_xgb_time: float = end_time_xgb - start_time_xgb;
print("Training time for XGB Regressor:", total_xgb_time);
## EXERCISE 2: Use the fitted models to make predictions on the test set.
## Measure prediction time for Random Forest
### Also, measure the time it takes for each model to make its predictions using the `time.time()` function to measure the times before and 
### after each model prediction.
start_time_pred_rf: float = time.time();
y_pred_rf = rf.predict(X_test);
end_time_pred_rf: float = time.time();
rf_pred_time: float = end_time_pred_rf - start_time_pred_rf;
print("Prediction time for Random Forrest Regressor:", rf_pred_time);
## Measure prediction time for Random Forest
start_time_pred_xgb: float = time.time();
y_pred_xgb = xgb.predict(X_test);
end_time_pred_xgb: float = time.time();
xgb_pred_time: float = end_time_pred_xgb - start_time_pred_xgb;
print("Prediction time for XGB Regressor:", xgb_pred_time);

## EXERCISE 3: Calculate the MEAN SQUARED ERROR and R^2 values for both models.
mse_rf: float = mean_squared_error(y_test, y_pred_rf);
mse_xgb: float = mean_squared_error(y_test, y_pred_xgb);
r2_rf: float = r2_score(y_test, y_pred_rf);
r2_xgb: float = r2_score(y_test, y_pred_xgb);
print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}');
print(f'Random Forest:  XGB = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}');

## EXERCISE 6: Calculate the standard deviation of the test data.
std_y = np.std(y_test);
print("Standard deviation of Y test", std_y);

# Visualise the results
plt.figure(figsize=(14, 6));

# Random Forest plot
plt.subplot(1, 2, 1);
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="blue",ec='k');
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model");
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev");
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, );
plt.ylim(0,6);
plt.title("Random Forest Predictions vs Actual");
plt.xlabel("Actual Values");
plt.ylabel("Predicted Values");
plt.legend();


# XGBoost plot
plt.subplot(1, 2, 2);
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="orange",ec='k');
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model");
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev");
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, );
plt.ylim(0,6);
plt.title("XGBoost Predictions vs Actual");
plt.xlabel("Actual Values");
plt.legend();
plt.tight_layout();
plt.show();