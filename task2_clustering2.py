#!/usr/bin/env python
# coding: utf-8

# # Task 2: Country Development Clustering Analysis
# 
# This script implements clustering analysis to group global nations into distinct socio-economic development tiers using K-Means, providing nuanced insights beyond simple "developing/developed" labels.

# In[50]:


#importing python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


# # Data Loading and Exploration

# In[51]:


#loading of data
df = pd.read_csv('Downloads/concepts and tech of AI/ASSESSMENT/datafiles/country_data.csv')


# In[52]:


# Display first few rows
df.head()


# In[53]:


# Checking data types and any missing values
df.info()
#check for missing values
df.isnull().sum()


# In[54]:


# Statistical summary
df.describe()


# # Data preprocessing

# In[55]:


#convert the country column which is a categorical data into numbers so the algorithm cam understand them

le = LabelEncoder()

labels = [df.columns[i] for i in [0]]

for l in labels:
    df[l] = le.fit_transform(df[l])

df


# # Model 1 - K-Means Clustering using Two Features and 3 K-clusters

# In[56]:


#Using Child Mortality and Income as these are the most highly differentiating variables, representing health crisis and economic prosperity, respectively. 
X = df.iloc[:, [1,5]].values

#Scaling the data so all features have similar ranges
# Standardization gives every indicator equal weight by transforming them to have a mean of 0 and SD of 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[57]:


# Apply K-Means with K=3 (representing the hypothesis of three general tiers)
kmeans = KMeans(n_clusters = 3, n_init = 'auto', random_state = 42)

kmeans.fit(X_scaled)
kmcenters = kmeans.cluster_centers_

centers = kmeans.cluster_centers_
print('Centers: \n', centers)


# In[58]:


#Plot the 2D Clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X_scaled), cmap='plasma', s=80)

# Set labels and title
plt.title(f'2D Clustering (K=3) using Child Mortality and Income')
plt.xlabel('Child Mortality')
plt.ylabel('Income (USD)')
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.grid(True)
plt.savefig('task2_2Dclustering.png', dpi=300, bbox_inches='tight')
plt.show()


# # Model 2 - K-Means Clustering using Three Features and 3 K-clusters

# In[59]:


#Adding Total Fertility, to see how the cluster boundaries shift. Total Fertility is a strong proxy for social development and women's education.
X = df.iloc[:, [1,5,8]].values

#Scaling the data so all features have similar ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[60]:


# Apply K-Means with K=3 (representing the hypothesis of three general tiers)
kmeans = KMeans(n_clusters = 3, n_init = 'auto', random_state = 42)

kmeans.fit(X_scaled)
kmcenters = kmeans.cluster_centers_

centers = kmeans.cluster_centers_
print('Centers: \n', centers)


# In[61]:


#Plot the 3D Clusters
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the data
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.predict(X_scaled), cmap='plasma', s=80, alpha=0.8)

# Set labels and title
ax.set_xlabel('Child Mortality')
ax.set_ylabel('Income (USD)')
ax.set_zlabel('Total Fertility')
ax.set_title(f'3D Clustering (K=3) with 3 Key Features')
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend1)
plt.savefig('task2_3Dclustering.png', dpi=300, bbox_inches='tight')
plt.show()
plt.tight_layout()


# # Model 3 - K-Means Clustering Using all Nine(9) Features and 5 K-clusters

# In[62]:


# Elbow method implementation (conceptual)
K_range = range(1, 11)
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)  # Replace 'data' with your actual dataset
    inertias.append(kmeans.inertia_)
    
# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('elbow_method_plot.png', dpi=300, bbox_inches='tight')
plt.show()
#Plot showed "elbow" (diminishing returns) at K=5
#Beyond K=5, additional clusters provided minimal inertia reduction


# In[63]:


#using all 9 features to find the optimal number of groups (K=5)
X = df.iloc[:, 1:].values

#Scaling the data so all features have similar ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[64]:


# Apply K-Means with K=3 (representing the hypothesis of three general tiers)
kmeans = KMeans(n_clusters = 5, n_init = 'auto', random_state = 42)

df['cluster'] = kmeans.fit(X_scaled)
kmcenters = kmeans.cluster_centers_

centers = kmeans.cluster_centers_
print('Centers: \n', centers)


# In[65]:


#Since using 9 features, visualization will be done using the top three raw features for a 3D plot.Child Mortality, Income, and Total Fertility


# In[66]:


#Plot the 3D Clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the data
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.predict(X_scaled), cmap='plasma', s=80, alpha=0.8)

# Set labels and title
ax.set_xlabel('Child Mortality')
ax.set_ylabel('Income (USD)')
ax.set_zlabel('Total Fertility')
ax.set_title(f'3D Clustering (K=5) with 3 Key Features')
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend1)
plt.savefig('task2_3Dclustering(K=5).png', dpi=300, bbox_inches='tight')
plt.show()
plt.tight_layout()


# In[67]:


# Get cluster centroids in original scale
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
cluster_summary = pd.DataFrame(centroids_original, columns=df.columns[:-1])
cluster_summary['cluster'] = range(5)

# Display summary
cluster_summary


# # CONCLUSION
# 
# This clustering analysis provided a powerful framework for understanding global development by progressing from a simple two-feature model to a comprehensive nine-feature, five-cluster segmentation. The initial model confirmed child mortality and income as core indicators of global inequality. The final K-Means model revealed nuanced tiers, successfully distinguishing between groups like "Developed and Stable Nations" and the "Extreme Poverty and Humanitarian Crisis" cluster. These precise profiles enable development agencies to move beyond generalized strategies and allocate resources more effectively, ensuring that interventions (e.g., humanitarian aid vs. institutional support) are targeted to the specific socio-economic needs of each country subgroup.

# In[ ]:




