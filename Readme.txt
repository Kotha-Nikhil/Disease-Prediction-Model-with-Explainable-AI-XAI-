# Disease Prediction Model with Explainable AI (XAI)

## **Overview**
This project builds a machine learning model for disease prediction using a **Random Forest Classifier** and interprets its predictions using **Explainable AI (XAI)** techniques with **SHAP (SHapley Additive exPlanations)**.

The goal is to predict disease outcomes based on input features and provide interpretable insights into how each feature influences the predictions. This notebook demonstrates the complete process, including data preprocessing, model training, evaluation, and interpretability analysis.

---

## **Table of Contents**
1. Introduction
2. Dataset Overview
3. Data Preprocessing
4. Model Training
5. Model Evaluation
6. Explainable AI (XAI) with SHAP
7. Conclusion and Next Steps

---

## **1. Introduction**
- **Objective**: Build a predictive model for disease classification and interpret the results using SHAP.
- **Techniques Used**:
  - Machine Learning with Random Forest.
  - Model interpretability using SHAP.
  - Visualizations for feature importance and decision-making.

---

## **2. Dataset Overview**
- **Dataset Source**: [Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)
- **Description**: Features include medical attributes such as glucose levels, BMI, age, and more, with a binary target variable indicating disease presence.
- **Key Features**:
  - Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
  - Outcome (Target Variable): 0 (No Disease) or 1 (Disease).

---

## **3. Data Preprocessing**
- Checked for missing values.
- Split data into **training** and **test sets**.
- Encoded categorical variables (if any).
- Scaled and normalized features where necessary.

---

## **4. Model Training**
- Used **Random Forest Classifier** with 100 estimators.
- Trained the model using **80% training data** and evaluated on the **20% test data**.

---

## **5. Model Evaluation**
- Measured performance using:
  - **Accuracy**.
  - **Confusion Matrix**.
  - **Classification Report** (Precision, Recall, F1-Score).
- Visualized confusion matrix for easy understanding of predictions.

---

## **6. Explainable AI (XAI) with SHAP**
- Used **SHAP KernelExplainer** for interpretability.
- Generated **summary plots** to highlight global feature importance.
- Created **force plots** to analyze individual predictions.

---

## **7. Conclusion and Next Steps**
- **Findings**:
  - The model achieved competitive accuracy and demonstrated clear interpretability with SHAP.
  - Identified key contributing features for disease prediction.
- **Future Improvements**:
  - Tune hyperparameters to optimize model performance.
  - Experiment with additional models (e.g., Gradient Boosting).
  - Test with larger, real-world datasets.

---

## **Requirements**
- Python 3.x
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - sklearn
  - shap

---

## **Usage**
1. Clone this repository.
2. Install the required libraries:  
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn shap
   ```
3. Run the notebook step-by-step.
4. Analyze results and visualizations generated in the outputs.

---

## **Contact**
For queries, contact: **Nikhil Kotha** at **kotha.n@northeastern.edu**.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

