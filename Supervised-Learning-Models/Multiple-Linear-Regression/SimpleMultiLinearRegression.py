import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

url: str = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv";

## Read the data from the csv file and understand it.
df: pd.DataFrame = pd.read_csv(url);
# print("Data obtained from CSV source");
# print("\n", df.sample(5));
# print("\n", df.describe());

## Drop any useless columns and categoricals that we don't need.
df: pd.DataFrame = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis = 1)
print("\n", df.corr());
# print("\n Reformatted data", df.describe());
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
print("\n", df.head(9));

## Now, plotting the graph for the correlation between 'FUELCONSUMPTION_COMB_MPG' and 'CO2EMISSIONS'
axes = pd.plotting.scatter_matrix(df, alpha=0.2);
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha("right")

plt.tight_layout();
plt.gcf().subplots_adjust(wspace=0, hspace=0);
plt.show();