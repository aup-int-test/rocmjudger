# Batch Normalization

## Description

Implement **Batch Normalization** in **FP32**.
Given an input matrix **X** of shape **[N, C]**, along with scale parameters **γ (gamma)** and bias parameters **β (beta)**, compute the normalized output matrix **Y** of the same shape.

For each channel **c**, the normalization is defined as:

$$
\mu_c = \frac{1}{N} \sum_{i=1}^{N} X_{i,c}
$$

$$
\sigma_c^2 = \frac{1}{N} \sum_{i=1}^{N} (X_{i,c} - \mu_c)^2
$$

$$
Y_{i,c} = \gamma_c \cdot \frac{X_{i,c} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c
$$

All computations use **32-bit floating point numbers (FP32)**.

---

## Input Description

The input consists of:

* Two integers **N** and **C**, representing the number of samples and channels.
* One floating-point number **ε (epsilon)** used for numerical stability.
* Followed by **N × C** floating-point numbers representing the input matrix **X** (in row-major order).
* Followed by **C** floating-point numbers representing the **γ (gamma)** parameters.
* Followed by **C** floating-point numbers representing the **β (beta)** parameters.

**Input format:**

```bash
N C ε
X_1 X_2 ... X_{N×C}
γ_1 γ_2 ... γ_C
β_1 β_2 ... β_C
```

---

## Output Description

Output **N × C** floating-point numbers representing the normalized result matrix **Y**, printed in row-major order.

Each row corresponds to one sample, and each value should be printed with space separation.

**Output format:**

```bash
Y_1_1 Y_1_2 ... Y_1_C
...
Y_N_1 Y_N_2 ... Y_N_C
```

---

## Constraints

```
1 ≤ N ≤ 10000
1 ≤ C ≤ 1024
All values are 32-bit floating point numbers.
```

---

## Example

**Input:**

```
3 2 1e-5
1.0 2.0
3.0 4.0
5.0 6.0
1.0 1.0
0.0 0.0
```

**Output:**

```
-1.224744 -1.224744
0.000000 0.000000
1.224744 1.224744
```

---