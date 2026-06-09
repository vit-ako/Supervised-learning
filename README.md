# Supervised-learning
Summary: This project focuses on supervised learning, particularly linear models, regularization techniques, overfitting, underfitting, and metrics for quality estimation.

I explored:

- analytical and gradient-based solutions for linear regression;
- the impact of L1 and L2 regularization on model weights;
- why L1 regularization performs feature selection;
- how linear models can model non-linear relationships;
- proper data preparation (normalization, polynomial features).

---

## 🔍 Theoretical Questions (Answers in Code & Analysis)

At the beginning of the project, I broke down key concepts:

### 1. Analytical Solution of Regression in Vector Form

Implemented in the `DeterministicLinearRegression` class using the `analytical` method.  
Uses the normal equation formula:  
**w = (XᵀX)⁻¹ Xᵀ y**

### 2. What Changes When Adding L1 and L2 Regularization

A penalty term is added to the loss function:
- **L2 (Ridge)** → `α * Σ w²`
- **L1 (Lasso)** → `α * Σ |w|`

This is implemented in `_compute_gradient` and for the analytical solution (for L2 — by modifying the XᵀX matrix).

### 3. Why L1 Sets Weights to Zero

I explained this in code comments:  
L1 creates a "linear penalty" where for unimportant features, it's more beneficial to set the weight to zero than to pay the penalty. This is **built-in feature selection**.

### 4. How to Model Non-linearities with Linear Methods

I demonstrated this in practice by adding **polynomial features** (squares of bathrooms, bedrooms, interest_level).  
Linear regression on these new features effectively models non-linear relationships.

---

## 🛠 Workflow & Implementation

### 1. Dataset Preparation

- Data: rental listings (price is the target variable).
- Outlier removal based on price (1st–99th percentiles).
- Extracted top-20 amenities (`features`) and created binary features.
- Built feature matrix X and target y.

### 2. From-Scratch Model Implementation

Created the `DeterministicLinearRegression` class supporting:

- analytical solution;
- gradient descent;
- stochastic gradient descent;
- L1, L2, and ElasticNet regularization.

### 3. Custom Normalization

- `MyMinMaxScaler`
- `MyStandardScaler`

**Key insight:** Normalization is needed when features have different scales (e.g., area vs. room count), but doesn't affect tree-based models.

### 4. Polynomial Features

I manually added squared features so the linear model could capture non-linear effects.

### 5. Evaluation Metrics

- MAE
- RMSE
- R²

**Comparison included:**
- my models vs. standard `sklearn` implementations;
- naive models (mean / median baseline).

---

## 📊 Results Summary

### Best Models by Test R² and Stability

| Model                 | Train R² | Test R²  | Difference |
|-----------------------|----------|----------|------------|
| MyLinReg_poly         | 0.50     | 0.514    | **0.014**  |
| ridge_poly            | 0.50     | 0.514    | **0.014**  |
| lasso_minmax          | 0.328    | 0.329    | **0.001**  |
| elastic_minmax        | 0.259    | 0.261    | 0.002      |

> ✅ **Conclusion**:  
> - Polynomial features produced the **most stable model** (minimal train/test gap).  
> - Normalization + L1/L2 significantly helps combat overfitting.  
> - My implementations match sklearn in quality (validating code correctness).

### Weakest Models

- Naive models (mean/median) — predictably poor performance.
- Some normalized models without polynomials — struggle with non-linearity.

---

## 🧠 What I Learned

✅ Implementing linear regression in multiple ways (analytical, GD, SGD).

✅ Understanding how regularization changes weights and why L1 performs feature selection.

✅ Observing the difference between L1 and L2 at the gradient and analytical solution levels.

✅ Creating non-linear features for linear models.

✅ Writing custom scalers (MinMax, Standard).

✅ Structuring model comparison results in DataFrames and drawing conclusions.

✅ Building a complete data preparation pipeline from scratch.

---

## 📂 Project Structure

- `DeterministicLinearRegression` — main model class
- `MyMinMaxScaler`, `MyStandardScaler` — custom normalizers
- Polynomial features generation
- Comparison with `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- MAE, RMSE, R² metrics
- Final results table

---

## 🚀 How to Run

1. Install dependencies: `pandas`, `numpy`, `scikit-learn`
2. Load `train.json` and `test.json`
3. Execute notebook cells sequentially

---

## 💡 Final Takeaway

This project transformed my understanding of machine learning from a "black box" approach to a deep, mathematical understanding. I no longer just call `.fit()` and `.predict()` — I know what happens inside. I'm especially proud of the L1/L2 implementations and the stability analysis of different models.
