# Concepts and Technologies of AI - Assessment
**Module:** 7CS030 | MSc Data Science | University of Wolverhampton  
**Tools:** Python, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

---

## 📋 Overview
This repository contains three machine learning tasks developed as part of 
the AI module assessment, covering supervised and unsupervised learning techniques 
applied to real-world datasets.

---

## 📁 Tasks

### Task 1 — House Price Prediction (Linear Regression)
**File:** `task1_regression.py`  
**Dataset:** King County, Washington housing sales (21,613 records)

Progressively built four regression models to predict house sale prices:
- Simple Linear Regression (1 feature: sqft_living) → R² = 0.50
- Multiple Regression (2 features: +grade) → R² = 0.55
- Multiple Regression (7 features) → R² = 0.64
- Comprehensive Model (all 18 features) → R² = 0.70

**Key finding:** Property quality (grade) and size (sqft_living) are the 
primary drivers of value. Year built and lot size had minimal impact.

---

### Task 2 — Country Development Clustering (K-Means)
**File:** `task2_clustering2.py`  
**Dataset:** Global socio-economic indicators (9 features across 167 countries)

Applied K-Means clustering in three stages:
- 2-feature model (child mortality + income, K=3) → baseline tiers
- 3-feature model (+ fertility rate, K=3) → social dynamics revealed
- 9-feature comprehensive model (K=5, Elbow Method) → five distinct clusters

**Clusters identified:**
| Cluster | Label |
|---------|-------|
| 0 | Developed & Stable Nations |
| 1 | Extreme Poverty & Humanitarian Crisis |
| 2 | Lower-Middle Income Nations |
| 3 | Least Developed but Transitioning |
| 4 | Emerging & Trade-Focused Economies |

---

### Task 3 — NBA Rookie Career Longevity Prediction (Classification)
**File:** `task3_Classification.py`  
**Dataset:** NBA rookie statistics (1,329 player records)

Compared three classifiers across three feature sets to predict whether 
a rookie will sustain a career of 5+ years:

| Model | Features | Logistic Regression | Naive Bayes | Neural Network |
|-------|----------|-------------------|-------------|----------------|
| Model 1 | 2 features | 0.74 | 0.71 | 0.74 |
| Model 2 | 3 features | 0.74 | 0.67 | **0.75** ✅ |
| Model 3 | All 19 | 0.73 | 0.59 | 0.72 |

**Best model:** Neural Network (40, 20 hidden layers) on 3 features  
→ Recall of 87.59% for identifying long-career players

---

## 🛠️ Libraries Used
```python
numpy | pandas | matplotlib | seaborn | scikit-learn
```

## ▶️ How to Run
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Run any task
python task1_regression.py
python task2_clustering2.py
python task3_Classification.py
```

> **Note:** Update the file paths in each script to point to your local 
> dataset files before running.
