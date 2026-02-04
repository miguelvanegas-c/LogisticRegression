# Heart Disease Prediction: Logistic Regression Analysis

This repository contains a comprehensive implementation of a **Logistic Regression** model developed from scratch to predict heart disease. The project focuses on manual algorithm implementation, optimization, visualization of decision boundaries, and regularization techniques to ensure model robustness.

## 📋 Exercise Summary

The workflow covers the complete Machine Learning lifecycle:

1.  **EDA & Prep:** Data cleaning, normalization, and train/test splitting.
2.  **Core Implementation:** Manual Gradient Descent and Cost Function (no `sklearn`).
3.  **Visualization:** 2D decision boundary analysis for clinical feature pairs.
4.  **Regularization:** Application of L2 (Ridge) penalty to prevent overfitting.

## 📊 Dataset Description

The analysis uses the **Kaggle Heart Disease Dataset**, containing clinical data from 303 patients:

- **Samples:** 303 patients.
- **Key Features:** Age, Sex, Cholesterol (112-564 mg/dL), Blood Pressure (BP), Max HR, ST Depression, and Number of Vessels.
- **Target Variable:** `Heart Disease` (Presence vs. Absence; ~55% prevalence).

---

## 🛠️ Project Development

### Step 1: Preprocessing and EDA

Initial data cleaning and feature engineering were performed, followed by a data split:

- **Train:** 70% for model optimization.
- **Test:** 30% for external validation.

### Step 2: Basic Logistic Regression

The optimization algorithm was implemented using:

- **Sigmoid Function:** $g(z) = \frac{1}{1 + e^{-z}}$.
- **Cost Function:** Binary Cross-Entropy.
- **Training:** 1500 iterations with a learning rate $\alpha = 0.01$.

**Initial Results:** The model shows stable convergence. Feature scaling (Standardization) was found to be critical for the Gradient Descent to reach the global minimum effectively.

### Step 3: Decision Boundary Visualization

We analyzed specific feature pairs to understand class separability:

- **Age vs. Cholesterol:** Shows significant overlap, indicating these are weak linear predictors on their own.
- **ST Depression vs. Number of Vessels:** Shows a clearer divide, where higher values in both categories drastically increase the probability of heart disease.

### Step 4: L2 Regularization (Ridge)

To control model complexity and prevent overfitting, an $L_2$ penalty was added to the cost function:
$$J(w,b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ \text{loss}(f_{w,b}(x^{(i)}), y^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=0}^{n-1} w_j^2$$

**Impact of $\lambda$:**

- **$\lambda = 0$:** Baseline model (prone to high variance).
- **$\lambda = 0.1$:** Identified as the optimal value, improving the **F1-Score** on the test set and successfully reducing weight magnitude ($||w||$).

---

## 🚀 Key Insights

1.  **Standardization:** Crucial for convergence; without it, features like `Cholesterol` dominate the gradient due to their scale.
2.  **Clinical Relevance:** Features such as `Thallium` and `Number of Vessels` showed the highest coefficients, aligning with medical importance.
3.  **Generalization:** Regularization smoothed the decision boundaries, allowing the model to ignore minor noise in the training data and perform better on unseen patients.

## 📦 Dependencies

- Python 
- NumPy
- Pandas
- Matplotlib
- Seaborn
