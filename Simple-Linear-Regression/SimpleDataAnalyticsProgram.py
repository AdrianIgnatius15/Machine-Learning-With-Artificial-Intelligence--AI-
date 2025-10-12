import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv";
df = pd.read_csv(url);

# Verify that the data has been loaded correctly
# print("First five rows of the dataset:");
# print(df.head());
# print("\nDataset information:");
# print(df.info());
# print("\nStatistical summary of the dataset:");
# print(df.describe());

# Select relevant features for the model
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']];
print("\nSelected features:");
print(cdf.head());
print("\nSample data:");
print(cdf.sample(9));

# Considering histogram of each feature
# viz = cdf[['CYLINDERS', 'ENGINESIZE', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']];
# viz.hist();
# plt.show();

# Plot the dataset to see the features against the CO2EMISSIONS and see their linear relationship
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue");
# plt.xlabel("Engine Size");
# plt.ylabel("CO2 Emissions");
# plt.xlim(0, 27);
# plt.show();

## PRACTICE
# Plot CYLINDERS vs CO2EMISSIONS
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="red");
plt.xlabel("Cylinders");
plt.ylabel("CO2 Emissions");
plt.xlim(0, 12);
plt.show();