# Supervised-learning
Summary: This project focuses on supervised learning, particularly linear models, regularization techniques, overfitting, underfitting, and metrics for quality estimation.

I explored:

- analytical and gradient-based solutions for linear regression;
- the impact of L1 and L2 regularization on model weights;
- why L1 regularization performs feature selection;
- how linear models can model non-linear relationships;
- proper data preparation (normalization, polynomial features).

---

# Theoretical Foundations: Linear Regression & Regularization

## 1. Analytical Solution for Regression Problems

**Find the analytical solution for a regression task using the vector form of the equation.**

The analytical (closed-form) solution for linear regression is derived by minimizing the Mean Squared Error (MSE) loss function:

**Vector form:**

```
w = (XᵀX)⁻¹ Xᵀ y
```

| Symbol | Meaning |
|--------|---------|
| `X` | Feature matrix |
| `y` | Target vector |
| `w` | Weight vector |

> **Note:** This solution works when `XᵀX` is invertible (features are not perfectly correlated).

---

## 2. Impact of L1 and L2 Regularization on the Solution

**What changes when adding L1 or L2 regularization to the loss function?**

A penalty term depending on the model parameters is simply added to the loss function:

### Comparison of Regularization Types

| Regularization | Penalty Term | Effect |
|:--------------|:-------------|:-------|
| **L2 (Ridge)** | `α Σ wᵢ²` | Shrinks weights toward zero (rarely exactly zero) |
| **L1 (Lasso)** | `α Σ \|wᵢ\|` | Shrinks weights to exactly zero → **feature selection** |
| **ElasticNet** | `α [ρ Σ \|wᵢ\| + (1-ρ) Σ wᵢ²]` | Combines both approaches |

> 💡 **Key insight:** Regularization creates a trade-off between fitting training data and keeping weights small, which prevents overfitting.

---

## 3. Why L1 Regularization for Feature Selection?

**Explain why L1 regularization is often used for feature selection. Why are many weights equal to zero after training?**

L1 regularization doesn't just penalize large weights (like L2 does) — it introduces a **linear penalty** proportional to the absolute value of each weight. This mechanism forces the model to "choose":

- ✅ **Feature stays** → if it contributes enough to the prediction to justify its penalty
- ❌ **Feature is removed** → if the contribution is too small, the weight becomes exactly zero

### Why L1 Yields Sparse Solutions

| L2 (Ridge) | L1 (Lasso) |
|:-----------|:-----------|
| Penalty is quadratic | Penalty is linear |
| Gradient goes to zero as weight → 0 | Constant gradient pushes weights to exactly zero |
| Weights become very small but rarely zero | Many weights become **exactly zero** |

**Result:** Unimportant features get zero coefficients — this is **automatic feature selection** built into the model.

---

## 4. Modeling Non-linear Relationships with Linear Models

**Explain how you can use the same models (linear regression, ridge regression, etc.) while capturing non-linear dependencies.**

Instead of changing the model itself, we transform the **feature space**:

```
Original features:   x₁, x₂, ..., xₙ
           ↓
New features:        z₁, z₂, ..., zₘ  (contain non-linearities)
           ↓
Linear model:        y = w₀ + w₁·z₁ + ... + wₘ·zₘ
```

### Examples of Non-linear Transformations

| Type | Formula | Use Case |
|:-----|:--------|:---------|
| Polynomial | `x²`, `x³`, `x₁·x₂` | Capturing curvature, interactions |
| Trigonometric | `sin(x)`, `cos(x)` | Periodic patterns |
| Exponential | `exp(x)` | Rapid growth/decay |
| Logarithmic | `log(x)` | Diminishing returns |
| Square root | `√x` | Slowing growth |

> 🎯 **Key takeaway:** A linear model applied to *non-linear features* becomes a non-linear model in the original feature space. This is the core idea behind **polynomial regression** and **basis function expansion**.

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
