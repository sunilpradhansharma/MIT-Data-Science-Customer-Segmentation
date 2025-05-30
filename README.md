
# 📊 Marketing Campaign Customer Segmentation

**MIT Applied Data Science Program - Capstone Project**  
**Author**: Sunil Pradhan Sharma  
**Date**: December 2024

---

## 🧠 Project Overview

The objective of this project is to segment customers based on demographic and behavioral data to enable **targeted marketing campaigns**, thereby increasing marketing efficiency, enhancing customer engagement, and improving return on investment (ROI). This project leverages clustering techniques such as **K-Means**, **DBSCAN**, and **Gaussian Mixture Models** to uncover meaningful customer segments.

---

## 🚩 Problem Statement

Most businesses use generic marketing strategies that don’t reflect customer preferences or behaviors. This leads to:
- Inefficient use of resources
- Poor customer engagement
- Low ROI

**Goal**: Use unsupervised learning to segment the customer base and develop personalized marketing strategies tailored to each segment.

---

## 🎯 Key Questions

- What defines each customer segment?
- How do customers differ in demographics and purchasing behavior?
- Which marketing channels are most effective per segment?
- What are the key drivers of customer purchasing decisions?
- Which customer groups offer the highest revenue potential?

---

## 📁 Dataset Summary

- **Rows**: 2,240 customers  
- **Columns**: 27 attributes (e.g., Income, Marital_Status, MntWines, NumWebPurchases)  
- **Missing Values**: 24 missing income values (imputed)
- **Data Types**: Mix of numerical, categorical, and date fields
- **Source**: Simulated marketing campaign data

---

## 🧹 Data Preprocessing

- Missing value imputation (Income)
- Outlier handling (Income, MntWines, MntGoldProds)
- Feature scaling (Standardization)
- Feature engineering:
  - `Age = 2024 - Year_Birth`
  - `Family_Members = 1 + Kidhome + Teenhome`
  - `Total_Spending = Sum(MntWines to MntGoldProds)`

---

## 🔍 Techniques Explored

| Technique         | Key Strengths                                  | Considerations                             |
|------------------|--------------------------------------------------|--------------------------------------------|
| K-Means           | Fast, interpretable, effective for spherical clusters | Sensitive to outliers, requires K          |
| K-Medoids         | More robust to outliers                         | Computationally heavier                    |
| DBSCAN            | No need to predefine clusters, handles outliers | Sensitive to hyperparameters               |
| Gaussian Mixture  | Probabilistic, flexible                         | Computationally expensive                  |
| PCA, t-SNE        | Used for dimensionality reduction and visualization | Not clustering algorithms themselves      |

---

## ✅ Final Model Choice: **DBSCAN**

**Reasons for Selection**:
- Handles arbitrary-shaped clusters
- Automatically identifies the number of clusters
- Robust to outliers and noisy data
- Aligns well with business objectives (e.g., isolating high spenders)

---

## 📊 Clustering Results

| Cluster     | Profile Description                                  |
|-------------|------------------------------------------------------|
| Cluster 0   | **Low-income, low-spending** — Price-sensitive customers |
| Cluster 1   | **High-income, high-spending** — VIPs preferring premium goods |
| Cluster 2   | **Moderate-income families** — Family-oriented buying behavior |
| Cluster -1  | **Outliers** — Irregular or unusual purchasing habits |

---

## 📢 Recommendations

- **Targeted Campaigns**:
  - 🎯 Cluster 0: Budget-friendly bundles and entry-level offers
  - 👪 Cluster 1: VIP promotions, exclusive catalogs, loyalty programs
  - 💼 Cluster 2: Family-pack offers and bundle discounts

- **Channel Optimization**:
  - Use web and email for digital-savvy users
  - Direct mail or offline for traditional customers

- **Customer Engagement**:
  - Retarget outliers with surveys or feedback offers
  - Personalize incentives based on recency and past behavior

---

## 📈 Business Impact

| Metric             | Projected Outcome                              |
|--------------------|-------------------------------------------------|
| 📈 Revenue Growth  | +15% to +25% from personalized campaigns        |
| 💵 ROI             | Doubled through precision targeting             |
| 🤝 Retention       | Enhanced via loyalty programs                   |
| 💡 Strategic Focus | Efficient budget allocation by cluster insights |

---

## 🚀 Next Steps

- Develop predictive models to forecast campaign effectiveness
- Automate segmentation and campaign orchestration via APIs
- Deploy interactive dashboards for marketing and product teams
- Integrate real-time data streaming (optional future scope)

---

## 🙏 Acknowledgment

This project was submitted as part of the **MIT Applied Data Science Program**, guided by expert instructors and industry mentors.

---

## 📬 Contact

**Sunil Pradhan Sharma**  
📧 sharmasunilpradhan@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sunil-p-sharma/)

---
