import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn import linear_model

df = pd.read_csv(r"F:\Coding Projects\AI\Machine Learning\Supervised-Learning-Models\new\Linear-Regression\Housing Prices.csv", encoding='latin1')
# print(df.head());

# plt.xlabel('Price (USD)');
# plt.ylabel("Square Feet");
# plt.scatter(df["Price"], df["Sq_Ft"], marker="+");
# plt.show();

linearReg = linear_model.LinearRegression();
linearReg.fit(df[["Sq_Ft"]], df["Price"]);

userInputForSquareFeet: str = input("Enter the square feet of the house you want to buy in Bucknell Rd, Costa Mesa, US: ");
prediction = linearReg.predict(pd.DataFrame({'Sq_Ft': [userInputForSquareFeet]}));
roundedPredictionValue : int = prediction[0];
print(f"The predicted price is: {roundedPredictionValue}");