# Ordinary Least Squares (OLS)

## Description

Solve the **Ordinary Least Squares (OLS)** regression problem on a GPU.
Given a feature matrix **X** of size **n_samples × n_features** and a target vector **y** of size **n_samples**,
compute the coefficient vector **β** that minimizes the sum of squared residuals:

$$
\min_{\beta} | X\beta - y |^2
$$

The closed-form solution to OLS is:

$$
\beta = (X^T X)^{-1} X^T y
$$

All matrices are stored in **row-major order** and use **32-bit floating point numbers (FP32)**.

---

## Input Description

The input consists of:

* Two integers **n_samples** and **n_features**,
  representing the number of samples and the number of features.
* Followed by **n_samples × n_features** floating-point numbers representing matrix **X** (row-major).
* Followed by **n_samples** floating-point numbers representing vector **y**.

**Input format:**

```bash
n_samples n_features
X_1 X_2 ... X_{n_samples×n_features}
y_1 y_2 ... y_{n_samples}
```

---

## Output Description

Output **n_features** floating-point numbers representing the coefficient vector **β**,
separated by spaces and ending with a newline.

**Output format:**

```bash
β_1 β_2 ... β_{n_features}
```

---

## Constraints

$$
1 \leq n_{\text{samples}} \leq 100{,}000
$$
$$
1 \leq n_{\text{features}} \leq 1{,}000
$$
$$
n_{\text{samples}} \geq n_{\text{features}}
$$
$$
-1000.0 \leq X_{ij}, y_i \leq 1000.0
$$

Solutions are tested with:

* Absolute tolerance ≤ **1e−2**
* Relative tolerance ≤ **1e−2**

---

## Example

**Input:**

```
3 2
1 2
2 3
3 4
1
2
3
```

**Output:**

```
0.000000 1.000000
```

---

## Implementation Requirements

* External libraries are **not permitted**.
* The `solve` function signature must remain unchanged.
* The final coefficients must be stored in the output vector **beta**.
* Assume that matrix **X** is **full rank**, i.e. ( X^T X ) is invertible.