"""
============================================================
K-Means Clustering
============================================================

This code shows how does 'K-Means' clustering in many data science applications. It is especially useful if you need to quickly discover
insights from unlabeled data.

Real-world application of 'K-Means' include:
 - Customer segmentation
 - Understandinngg what website visitors are trying to accomplish
 - Pattern recognition
 - Feature engineering
 - Data compression
"""

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.cluster import KMeans;
from sklearn.datasets import make_blobs;
from sklearn.preprocessing import StandardScaler, OneHotEncoder;
import plotly.express as px;
import seaborn as sns;
import warnings;
warnings.filterwarnings("ignore");

'''
===============================
K-Means on a synthetic data set
===============================

We will create our own dataset.

First, we need to set a random seed. Use `np.random.seed(0)` function, where the seed will be `0`.
'''
# np.random.seed(0);

'''
Next, we will be making random clusters of points using `make_blobs` class. The `make_blobs` class can take in many inputs, but we will
be using the specific ones:

Input

1. n_samples: The total number of points equally divided among clusters.
  - Value will be: 5000
2. centres : The number of centres to generate, or the fixed centre locations.
  - Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
3. cluster_std: The standard deviation of the clusters.
  - Value will be: 0.9

Output
1 X: Array of shape [n_samples, n_features]. (Feature Matrix)
  - The generated samples.
2 y: Array of shape [n_samples]. (Response Vector)
  - The integer labels for cluster membership of each sample.
'''

# X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3],[1,1]], cluster_std=0.9);

'''
Display the scatter plot of the randomly generated data.
'''
# plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.3, ec='k', s=80);

'''
===============================
Setting up K-Means
===============================

Now that we have our random data, let's set up our k-means Clustering.
The KMeans class has many parameters that can be used, but we will be using these three:

init: Initialization method of the centroids.
- Value will be: k-means++
- k-means++: Selects initial cluster centres for k-means clustering in a smart way to speed up convergence.
n_clusters: The number of clusters to form as well as the number of centroids to generate.
- Value will be: 4 (since we have 4 centres)
n_init: Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
- Value will be: 12
Initialize KMeans with these parameters, where the output variable is called k_means.
'''
# k_means = KMeans(init = 'k-means++', n_clusters = 5, n_init = 12);
# k_means.fit(X);
# k_means_labels = k_means.labels_;
# k_means_cluster_centers = k_means.cluster_centers_;

'''
===============================
Creating the Visual Plot
===============================

Now that we have the random data generated and the 'K-Means' model initialised, let's plot the result and see what it looks like!
Please read through the code and code comments to understand how to plot the model.
'''
# # Initialize the plot with the specified dimensions.
# fig = plt.figure(figsize=(6, 4))

# # Colors uses a color map, which will produce an array of colors based on
# # the number of labels there are. We use set(k_means_labels) to get the
# # unique labels.
# colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

# # Create a plot
# ax = fig.add_subplot(1, 1, 1)

# # For loop that plots the data points and centroids.
# # k will range from 0-3, which will match the possible clusters that each
# # data point is in.
# for k, col in zip(range(len(k_means.cluster_centers_)), colors):

#     # Create a list of all data points, where the data points that are 
#     # in the cluster (ex. cluster 0) are labeled as true, else they are
#     # labeled as false.
#     my_members = (k_means_labels == k)

#     # Define the centroid, or cluster center.
#     cluster_center = k_means_cluster_centers[k]

#     # Plots the datapoints with color col.
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

#     # Plots the centroids with specified color, but with a darker outline
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# # Title of the plot
# ax.set_title('KMeans')

# # Remove x-axis ticks
# ax.set_xticks(())

# # Remove y-axis ticks
# ax.set_yticks(())

# # Show the plot
# plt.show()

'''
===============================
Customer Segmentation with K-Means
===============================

Imagine that you have a customer dataset, and you need to apply customer segmentation to this historical data. 
Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. 
It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources. 
For example, one group might contain customers who are high-profit and low-risk, or more likely to purchase products, or subscribe to a service. 
A business task is to retain those customers.
'''
# Load the data from the URL
## We will use the 'Cust_Segmentation.csv' dataset from the IBM Cloud Object Storage.
cust_df: pd.DataFrame = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv");
print("Initial look on the data:\n", cust_df.head());

# Preprocessing
## As we can see, the `Address` column is a categorical data type and it will not work with K-Means because the K-Means works with numerical data.
## Hence we can either one-hot code it or drop it.
# categorical_columns = cust_df.select_dtypes(include=['object']).columns.tolist();
# encoder: OneHotEncoder = OneHotEncoder(sparse_output=False, drop="first");
# encoder_features = encoder.fit_transform(cust_df[categorical_columns]);
# encoded_df: pd.DataFrame = pd.DataFrame(encoder_features, columns=encoder.get_feature_names_out(categorical_columns));
# cust_df: pd.DataFrame = pd.concat([cust_df.drop(columns=categorical_columns), encoded_df], axis=1);
cust_df = cust_df.drop(['Address'], axis=1);
print("\nData after dropping the Address column:\n", cust_df.head());

# Normalising over the standard deviation
## Now let's normalize the dataset. 
## But why do we need normalization in the first place? 
## Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally, 
## tranforming the features so they have zero mean and standard deviation of one. 
## We use StandardScaler() to normalize, or standardize our dataset.
X = cust_df.values[:, 1];
cluster_dataset = StandardScaler().fit_transform(X);

# Modeling
## We now apply the K-Means to the normalised dataset.
k_means: KMeans = KMeans(init="k-means++", n_clusters=3, n_init=12);
k_means.fit(X);
labels = k_means.labels_;
print("\nCluster labels:\n", labels);

# Insights
## We can now add the cluster labels to the original dataset.
cust_df["Clus_km"] = labels;
cust_df.groupby("Clus_km").mean();
area = np.pi * ( X[:, 1])**2;
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k',alpha=0.5);
plt.xlabel('Age', fontsize=18);
plt.ylabel('Income', fontsize=16);
plt.show();

# Create interactive 3D scatter plot
fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float));

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ));  # Remove color bar, resize plot

fig.show();