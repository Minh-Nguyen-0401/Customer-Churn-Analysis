# Customer Churn Prediction Project

---

## Project Overview

This project focuses on developing predictive models to assess the likelihood of customers leaving a business. By understanding and addressing customer attrition, businesses in industries such as telecommunications, internet service providers, pay TV companies, insurance firms, and alarm monitoring services can implement effective retention strategies.

### Problem Statement

Customer churn prediction is crucial for business management in sectors like telecommunications, internet service providers, pay TV companies, insurance firms, and alarm monitoring services. It involves understanding and addressing customer attrition—the loss of clients or customers. Retaining existing customers is significantly more cost-effective than acquiring new ones, making churn prediction essential for prioritizing retention efforts and maximizing long-term value.

### Key Questions

1. **What is the significance of Churn Rate for stakeholders (Customers, MCI, etc.)?**
2. **What are the characteristics of each Type of Customer (Churn or Not Churn)?**
3. **Which ML models can be implemented and how do they perform?** Including feature inputs and importance.
4. **What qualitative and quantitative actions could enhance retention rates?**

### Keywords

Binary Classification, Python, EDA, Statistical Significance

---

## Project Structure

- **`submission_workspace/`**: Folder containing the codebase, dataset, reference documents, and other project-related files.
- **`churn_analysis_workspace.ipynb`**: Main Jupyter Notebook for analysis, quick insights, and detailed end-to-end process flow of the project.
- **`churn_analysis_case_deck_final.pdf`**: Pitch deck (Slideshow) meeting business case requirements, exported in PDF format.
- **Other Files**: Reference documents and exported datasets relevant to the project.

## About the Dataset

- **State**: Two-letter code representing the customer's residence.
- **Account length**: Number of days with the current telecom provider.
- **Area code**: Area code of the customer's phone number.
- **International plan**: Indicates if the customer has an international calling plan (Yes/No).
- **Voice mail plan**: Indicates if the customer has a voicemail plan (Yes/No).
- **Number vmail messages**: Number of voicemail messages the customer has.
- **Total day minutes**: Total daytime minutes used.
- **Total day calls**: Total daytime calls made.
- **Total day charge**: Charges for daytime usage.
- **Total eve minutes**: Total evening minutes used.
- **Total eve calls**: Total evening calls made.
- **Total eve charge**: Charges for evening usage.
- **Total night minutes**: Total nighttime minutes used.
- **Total night calls**: Total nighttime calls made.
- **Total night charge**: Charges for nighttime usage.
- **Total intl minutes**: Total international minutes used.
- **Total intl calls**: Total international calls made.
- **Total intl charge**: Charges for international usage.
- **Customer service calls**: Number of calls to customer service.
- **Churn**: Indicates if the customer has churned (Yes/No).


## Workflow Outline

### 1. Introduction
|___ 1.1 Problem Statement  
|___ 1.2 Key Questions  
|___ 1.3 Requirements  
|___ 1.4 Keywords  

### 2. Data Collection and Understanding

### 3. Data Preprocessing
|___ 3.1 Column Reordering  
|___ 3.2 Standardize Column Names  
|___ 3.3 Modify Data Types  
|___ &nbsp;&nbsp;&nbsp;&nbsp;*Quick Comment*  
|___ 3.4 Check Outliers  

### 4. Exploratory Data Analysis (EDA)
|___ 4.1 Univariate Analysis  
|___ 4.2 Comparative Analysis Between Churned and Non-Churned Customers  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ 4.2.1 Overview  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ 4.2.2 Bivariate Analysis  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ a. Demographics & Tenure  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ b. Service Plans & Offerings  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ Key Insights - Intl Plan & Voicemail  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ Key Insights - International Calls  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ Key Insights - Voicemail Usage  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ c. Usage Patterns  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ d. Customer Service Interaction  

### 5. Feature Engineering
|___ 5.1 Review Attributes  
|___ 5.2 Discretize Attributes  
|___ 5.3 Normalize Features *(for XGBoost, LightGBM)*  
|___ 5.4 One-Hot Encode Categorical Attributes  
|___ 5.5 Feature Evaluation & Selection  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ 5.5.1 Correlation Matrix  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ Address Multicollinearity  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ 5.5.2 Information Value Model  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ IV Score Reference  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|___ Quick Insights  

### 6. Model Selection
|___ Overview - Model Pipeline  
|___ 6.1 Upsampling Necessity  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ Choosing LightGBM & XGBoost  
|___ 6.2 Load Train & Holdout Datasets  
|___ 6.3 Fit Models with CV Grid Search  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ a. Fit Dataset Version 1  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ b. Fit Dataset Version 2  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ c. Compare Models  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ d. Conclusion - LightGBM Selected  
|___ 6.4 Feature Importance  
|&nbsp;&nbsp;&nbsp;&nbsp;|___ Key Takeaways  

### 7. Final Model Deployment & Interpretation
|___ **Key Insights:** Feature Importance by Key Aspects (Model & EDA)  

---

## Machine Learning Model Development Pipeline

After refining features during Feature Engineering, this section outlines the steps to build and deploy the Machine Learning Model:

1. **Shortlist ML Models**: Utilize Ensemble Learning Models like XGBoost and LightGBM to address class imbalance and prevent overfitting.
2. **Assess Resampling Needs**: Consider SMOTE upsampling if necessary.
3. **Generate Dataset Versions**: Create two separate training and holdout datasets to evaluate specific features (`voice_mail_plan`, `intl_charge_per_min_cate`, `total_eve_calls`, `account_length`).
4. **Tune Hyperparameters with Grid Search CV**: Implement Grid Search for efficient hyperparameter tuning.
5. **Fit Optimized Models**: Train models on each dataset version.
6. **Compare Performance Metrics**: Evaluate using ROC AUC or precision-recall due to class imbalance.
7. **Adjust Features for Deployment**: Refine the final feature set.
8. **Final Model Implementation with BayesSearch CV & Interpretation**: Optimize hyperparameters with Bayes Search CV for final deployment.

---

**UPDATED**: Stacking Ensemble Model with top 3 base models (CatBoostClassifier, XGBoostClassifier, LGBMClassifier) generated relatively BETTER results
<br>

/rarr: resort to **`ensemble model.ipynb`** for more details

---
