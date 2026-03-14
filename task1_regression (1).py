#!/usr/bin/env python
# coding: utf-8

# 
# # Task 1: House Price Prediction using Linear Regression
# 
# This script implements the Linear Regression and aims to build a model that predicts house sale price based on property features.
# 
# The core technique is Linear Regression, which models a straight-line relationship between features (like size) and the target (price).

# In[43]:


#importing python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# # Data Loading and Exploration

# In[44]:


#loading and exploration of data
df = pd.read_csv('Downloads/concepts and tech of AI/ASSESSMENT/datafiles/houseprice_data.csv')


# In[45]:


# Display first few rows
df.head()


# In[46]:


# Checking the data types 
df.info()


# In[47]:


#check for missing values
df.isnull().sum()


# In[48]:


#the statistical summary
df.describe()


# # Correlation Analysis

# In[49]:


#other features correlation with price
corr = df.corr()
corr.style.background_gradient(cmap = 'coolwarm')


# In[50]:


correlation_matrix = df.corr()
price_corr = correlation_matrix['price'].sort_values(ascending=False)

price_corr.head(10)


# # MODEL 1 - SIMPLE LINEAR REGRESSION WITH ONE FEATURE
# 
# Simple Linear Regression helps establish a baseline performance using the single best predictor.
# 
# Feature Selection: 'sqft_living' (square footage) showed the highest correlation (r=0.70) with price.

# In[51]:


#starting with the most correlated (sqft_living)
X = df.iloc[:, [3]].values
y = df.iloc[:, 0].values


# In[52]:


#split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# In[53]:


#create and train the model
model1 = LinearRegression()
model1.fit(X_train, y_train)


# In[54]:


#evaluate the performance 
# R-squared measures the percentage of price variation explained by the model.
print('Coefficients: ', model1.coef_)
print('Intercept: ', model1.intercept_)
print('Mean Squared error: %.8f' %mean_squared_error(y_test, model1.predict(X_test)))
print('Coefficient of determination: %.2f' %r2_score(y_test, model1.predict(X_test)))


# In[55]:


y_pred = model1.predict(X_test)


# In[56]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
# Plot 1: Regression line with data points
axes[0].scatter(X_test, y_test, color = 'blue', alpha=0.5, label='Actual Prices')
axes[0].plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
axes[0].set_xlabel('sqft_living', fontsize=12)
axes[0].set_ylabel('Price ($)', fontsize=12)
axes[0].set_title(f'Simple Linear Regression: sqft_living vs Price', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
    
# Plot 2: Actual vs Predicted
axes[1].scatter(y_test, y_pred, color = 'blue', alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Price ($)', fontsize=12)
axes[1].set_ylabel('Predicted Price ($)', fontsize=12)
axes[1].set_title('Actual vs Predicted Prices', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
    
plt.tight_layout()
plt.savefig('task1_simple_regression.png', dpi=300, bbox_inches='tight')
plt.show()


# # MODEL 2 - MULTIPLE LINEAR REGRESSION USING TWO FEATURES
# 
# Multiple Linear Regression improves prediction accuracy by adding a set of strong, correlated features.
# 
# Features Added: 'grade' (overall quality) is the second strongest predictor (r=0.67).

# In[57]:


#MULTIPLE REGRESSION ON 2 FEATURES (sqft_living and grade)
X = df.iloc[:, [3, 9]].values
y = df.iloc[:, 0].values


# In[58]:


#split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 40)


# In[59]:


#train the model
model2 = LinearRegression()
model2.fit(X_train, y_train)


# In[60]:


#evaluate the performance
# R-squared measures the percentage of price variation explained by the model.
print('Coefficients: ', model2.coef_)
print('Intercept: ', model2.intercept_)
print('Mean Squared error: %.8f' %mean_squared_error(y_test, model2.predict(X_test)))
print('Coefficient of determination: %.2f' %r2_score(y_test, model2.predict(X_test)))


# In[61]:


fig1 = plt.figure(figsize=(13,7))
ax1 = fig1.add_subplot(111, projection = '3d')

ax1.scatter(X_train[:,0], X_train[:,1], y_train, color = 'blue', label = 'Prices')

x1_vals = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 30)
x2_vals = np.linspace(X_train[:,1].min(), X_train[:,1].max(), 30)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = model2.coef_[0]*X1 + model2.coef_[1]*X2 + model2.intercept_
ax1.plot_surface(X1, X2, Z, alpha=0.5, color = 'red', label = 'Regression plane')

ax1.azim = -60
ax1.dist = 10
ax1.elev = 20

# Limits 
ax1.set_xlim(X_train[:,0].min(), X_train[:,0].max())
ax1.set_ylim(X_train[:,1].min(), X_train[:,1].max())
ax1.set_zlim(y_train.min(), y_train.max())

ax1.set_xlabel('Soft_living',fontsize=12)
ax1.set_ylabel('Grade', fontsize=12)
ax1.set_zlabel('Price ($)', fontsize=12)
ax1.set_title(f'Mulitiple Regression: sqft_living & Grade vs Price', fontsize=14)
fig1.savefig('task1_Multiple_regression_3D_plot.png')

fig1.tight_layout()
ax1.legend()


# # MULTIPLE REGRESSION ON More FEATURES

# In[62]:


#multiple regression using sqft_living, view,grade,sqft_above,sqft_basement,yr_built,sqft_living15
X = df.iloc[:, [3, 7, 9, 10, 11, 12, 17]].values
y = df.iloc[:, 0].values


# In[63]:


#split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 40)


# In[64]:


#train the model
model3 = LinearRegression()
model3.fit(X_train, y_train)


# In[65]:


#evaluate the performance
# R-squared measures the percentage of price variation explained by the model.
print('Coefficients: ', model3.coef_)
print('Intercept: ', model3.intercept_)
print('Mean Squared error: %.8f' %mean_squared_error(y_test, model3.predict(X_test)))
print('Coefficient of determination: %.2f' %r2_score(y_test, model3.predict(X_test)))


# # Comprehensive Model - MULTIPLE REGRESSION ON all FEATURES
# 
# To test if adding ALL available features maximizes predictive accuracy.

# In[66]:


X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


# In[67]:


#split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)


# In[68]:


#train the model
model4 = LinearRegression()
model4.fit(X_train, y_train)


# In[69]:


#evaluate the performance
# R-squared measures the percentage of price variation explained by the model.
print('Coefficients: ', model4.coef_)
print('Intercept: ', model4.intercept_)
print('Mean Squared error: %.8f' %mean_squared_error(y_test, model4.predict(X_test)))
print('Coefficient of determination: %.2f' %r2_score(y_test, model4.predict(X_test)))


# # CONCLUSION
# 
# This analysis successfully developed a Multiple Linear Regression model that predicts house prices in King County with reasonable accuracy, ultimately explaining 70% of the price variance (R^2=0.70) across the dataset. The systematic progression from a simple baseline model (R^2=0.50) to a complex one validated the value of incorporating diverse features. The model confirms that property quality (grade), size (sqft\_living), and neighborhood context are the primary drivers of value, while variables often assumed to be critical, such as year built and lot size, had a minimal individual impact. These findings provide clear guidance for investors and homeowners: focusing resources on quality upgrades and enhancing view amenities offers the most significant leverage to increase property value
