#!/usr/bin/env python
# coding: utf-8

# # NBA Rookie Career Longevity Prediction
# 
# This script implements a comprehensive machine learning pipeline to predict whether NBA rookies will sustain careers of 5+ years based on their first-year statistics.
# 
# Three algorithms are evaluated:
# - Logistic Regression (baseline linear model)
# - Gaussian Naive Bayes (probabilistic classifier)
# - Neural Network/MLP (non-linear pattern detector)

# In[76]:


#importing python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')


# # Data Loading and Exploration

# In[77]:


#loading of dataset
df = pd.read_csv('Downloads/concepts and tech of AI/ASSESSMENT/datafiles/nba_rookie_data.csv')


# In[78]:


# Display first few rows
df.head()


# In[79]:


# Checking data types and any missing values
df.info()
print(df.isnull().sum())


# In[80]:


# Remove missing values
df = df.dropna()

# Drop the 'Name' column as it has no predictive value
# Player names are identifiers, not performance metrics
if 'Name' in df.columns:
    df = df.drop(columns=['Name'])

# Move target variable to first column for convenience
# This makes it easier to separate features from target
cols = df.columns.tolist()
new_order = [cols[-1]] + cols[:-1]
df = df[new_order]

df.info()


# In[81]:


# Analyze the distribution of our target variable
target_dist = df.iloc[:, 0].value_counts()
target_pct = df.iloc[:, 0].value_counts(normalize=True) * 100

print(f"\nTarget Variable: {0}")
print(f"  0 (Career < 5 years): {target_dist[0]} players ({target_pct[0]:.2f}%)")
print(f"  1 (Career ≥ 5 years): {target_dist[1]} players ({target_pct[1]:.2f}%)")
print(f"\nClass Balance: {target_pct[1]:.2f}% positive class")
print("  → Moderate imbalance (62% positive), workable for classification")


# In[82]:


# Statistical summary -basic statistics for all features
df.describe()


# # Correlation Analysis: Feature Selection
# The goal here is to determine which rookie statistics are most indicative of a career lasting $\ge 5$ years ($\text{TARGET\_5Yrs}=1$). We focus on the top four positive correlations to select features for our models.

# In[83]:


## Calculate correlation between each feature and the target variable
# Higher absolute correlation = stronger predictive relationship

corr = df.select_dtypes(include='number').corr()
corr.style.background_gradient(cmap='coolwarm')


# In[84]:


# Compute correlation of each feature with the target variable
correlations = df.corr()['TARGET_5Yrs'].sort_values(ascending=False)

# Display top 10 features most correlated with the target
print("\nTop 10 features correlated with Target_5Yrs:")
correlations.head(10)


# The strongest predictor is Games Played, suggesting that a player's initial ability to stay healthy and earn floor time is the single most important factor. The top three features chosen for Model 2 (Games Played, Points Per Game, Field Goals made) were selected due to these strong initial correlations.

# # Preparing data for modeling

# In[88]:


# Separate features (X) from target (y)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]


# In[89]:


# Split data into training (67%) and testing (33%) sets
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# In[90]:


# Initialize StandardScaler for feature normalization
# Scaling ensures all features contribute equally (important for Neural Networks)
scaler = StandardScaler()


# # MODEL1 - using two features

# In[91]:


#selecting the correlated features: games played and field goals made
X = df.iloc[:, [1,4]].values
y = df.iloc[:, 0].values


# In[92]:


#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[93]:


# Split data into training (67%) and testing (33%) sets
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)


# Logistic Regression

# In[94]:


LR_model1 = LogisticRegression(max_iter=1000, random_state=42)
LR_model1.fit(X_train, y_train)


# In[95]:


print('Our accuracy is %.2f' %LR_model1.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != LR_model1.predict(X_test)).sum()))


# In[96]:


ConfusionMatrixDisplay.from_predictions(y_test, LR_model1.predict(X_test))


# Naives bayes

# In[97]:


NB_model1 = GaussianNB()
NB_model1.fit(X_train, y_train)


# In[98]:


print('Our accuracy is %.2f' %NB_model1.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != NB_model1.predict(X_test)).sum()))


# In[99]:


ConfusionMatrixDisplay.from_predictions(y_test, NB_model1.predict(X_test))


# 
# 
# Neural Network

# In[100]:


NN_model1 = MLPClassifier(hidden_layer_sizes = (10,10), activation = 'relu', max_iter = 2000, random_state = 0)
NN_model1.fit(X_train, y_train)


# In[101]:


print('Our accuracy is %.2f' %NN_model1.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != NN_model1.predict(X_test)).sum()))


# In[102]:


ConfusionMatrixDisplay.from_predictions(y_test, NN_model1.predict(X_test))


# In[103]:


# Create mesh grid over feature space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Define models and titles
models = {
    'Logistic Regression': LR_model1,
    'Naive Bayes': NB_model1,
    'Neural Network': NN_model1
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
model_names = ['Logistic Regression', 'Naive Bayes', 'Neural Network']
models = [LR_model1, NB_model1, NN_model1]

for ax, name, model in zip(axes, model_names, models):
    if name == 'Neural Network':
        Z = model.predict_proba(grid)[:, 1]
        Z = (Z > 0.5).astype(int)
    else:
        Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='red', label='< 5 years', edgecolors='k', s=60)
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='green', label='≥ 5 years', edgecolors='k', s=60)
    mis_ind = np.where(y_test != model.predict(X_test))
    ax.scatter(X_test[mis_ind, 0], X_test[mis_ind, 1], marker = '*', color = 'white', s=3)


    ax.set_title(f"{name} Decision Boundary")
    ax.set_xlabel(df.columns[1])
    ax.set_ylabel(df.columns[4])
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig('task3_Classification.png', dpi=300, bbox_inches='tight')
plt.show()



# # MODEL2 - using three FEATURES

# In[104]:


#selecting the correlated features: games played, fields goal made and points per game
X = df.iloc[:, [1, 3, 4]].values
y = df.iloc[:, 0].values

#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[105]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)


# Logistic Regression

# In[106]:


LR_model2 = LogisticRegression(max_iter=1000, random_state=42)
LR_model2.fit(X_train, y_train)


# In[107]:


print('Our accuracy is %.2f' %LR_model2.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != LR_model2.predict(X_test)).sum()))


# In[108]:


ConfusionMatrixDisplay.from_predictions(y_test, LR_model2.predict(X_test))


# NAIVE BAYES 

# In[109]:


NB_model2 = GaussianNB()
NB_model2.fit(X_train, y_train)


# In[110]:


print('Our accuracy is %.2f' %NB_model2.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != NB_model2.predict(X_test)).sum()))


# In[111]:


ConfusionMatrixDisplay.from_predictions(y_test, NB_model2.predict(X_test))


# NEURAL NETWORKS

# In[112]:


NN_model2 = MLPClassifier(hidden_layer_sizes = (40, 20), activation = 'relu', max_iter = 2000, random_state = 42)
NN_model2.fit(X_train, y_train)


# In[113]:


print('Our accuracy is %.2f' %NN_model2.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != NN_model2.predict(X_test)).sum()))


# In[114]:


print(classification_report(y_test, NN_model2.predict(X_test), target_names=['< 5 years (0)', '>= 5 years (1)'], digits=4))


# In[115]:


ConfusionMatrixDisplay.from_predictions(y_test, NN_model2.predict(X_test))


# # MODEL3 -  USING ALL FEATURES

# In[118]:


#selecting all features
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[119]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42)


# LOGISTIC REGRESSION

# In[120]:


LR_model3 = LogisticRegression(max_iter=1000, random_state=42)
LR_model3.fit(X_train, y_train)


# In[121]:


print('Our accuracy is %.2f' %LR_model3.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != LR_model3.predict(X_test)).sum()))


# In[122]:


ConfusionMatrixDisplay.from_predictions(y_test, LR_model3.predict(X_test))


# NAIVES BAYES 

# In[123]:


NB_model3 = GaussianNB()
NB_model3.fit(X_train, y_train)


# In[124]:


print('Our accuracy is %.2f' %NB_model3.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != NB_model3.predict(X_test)).sum()))


# In[125]:


ConfusionMatrixDisplay.from_predictions(y_test, NB_model3.predict(X_test))


# NEURAL NETWORK 

# In[126]:


NN_model3 = MLPClassifier(hidden_layer_sizes = (10, 10, 10), activation = 'relu', max_iter = 2000, random_state = 42)
NN_model3.fit(X_train, y_train)


# In[127]:


print('Our accuracy is %.2f' %NN_model3.score(X_test, y_test))
print('Number of mislabeled points out of a total %d points : %d'
     %(X_test.shape[0], (y_test != NN_model3.predict(X_test)).sum()))


# In[128]:


ConfusionMatrixDisplay.from_predictions(y_test, NN_model3.predict(X_test))


# # Final Model Analysis and Conclusion
# 
# Model Effectiveness and Key Metrics
# The Neural Network (Model 2) achieved the highest overall performance. By analyzing the classification report, we gain deeper insight than simple accuracy:
# 
# precision <5 years(0)      0.6814,
# 
# precision >=5 years(1)     0.7697
# 
# recall <5 years(0)         0.5033,
# 
# recall >=5 years(1)        0.8759
# 
# f1_score <5 years(0)       0.5789, 
# 
# f1_score >=5 years(1)      0.8194,
# 
# accuracy 0.7472
# 
# Interpretation:
# 
# High Recall for Class 1 (0.8759): This is excellent. The model correctly identifies 87.59% of players who actually went on to have a long career. In a scouting context, this is critical, as it minimizes False Negatives (failing to identify a star player).
# 
# Precision vs. Recall: The Precision for Class 1 is slightly lower (0.7697). This means that out of all the players the model predicts will succeed, about 24% will actually fail (False Positives).
# 
# Class 0 Performance: Performance for the < 5$ years class is notably weaker, particularly the Recall (0.5033). The model often misclassifies shorter-career players as long-career players, which is typical when training data favors the majority class (Class 1).

# # CONCLUSION
# 
# The best model is a Neural Network (40, 20 hidden layers) trained on Games Played, Points Per Game, and Field Goals Made. It achieves a balance of high accuracy and high recall for the desired outcome ($\ge 5$ years), confirming that early volume and basic efficiency are strong indicators of career longevity. The project highlights that more data is not always better; thoughtful feature selection is critical for model success.

# In[ ]:




