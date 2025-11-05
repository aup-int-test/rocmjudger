# Logistic Regression

## Description

Solve the **logistic regression** problem on a GPU using HIP.
Given a feature matrix **X** of size `n_samples × n_features` and a binary target vector **y** of size `n_samples` (containing only 0s and 1s), compute the coefficient vector **β** that maximizes the log-likelihood:

[
\max_\beta \sum_{i=1}^{n} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]
]

where ( p_i = \sigma(X_i^T \beta) ) and ( \sigma(z) = \frac{1}{1 + e^{-z}} ).

Your implementation should use **Newton’s Method** to iteratively update β until convergence.
At each iteration, compute the gradient and Hessian matrix using HIP kernels, then solve for the Newton step on the CPU.

* **External libraries are not permitted**
* The `solve` function signature must remain unchanged
* The final coefficients must be stored in the **β** vector

## Input Description

The input specifies the dataset dimensions followed by matrix `X` and vector `y`.

Input format:

```bash
n_samples n_features
X_11 X_12 ... X_1n
X_21 X_22 ... X_2n
...
X_m1 X_m2 ... X_mn
y_1 y_2 ... y_m
```

Constraints:

* 1 ≤ n_samples ≤ 100,000
* 1 ≤ n_features ≤ 1,000
* n_samples ≥ n_features
* −10.0 ≤ values in X ≤ 10.0
* y ∈ {0, 1}

Solutions are tested with:

* Absolute tolerance ≤ 1e−2
* Relative tolerance ≤ 1e−2

## Output Description

Output the learned coefficient vector **β** containing `n_features` floating-point values, separated by spaces, followed by a newline.

Output format:

```bash
β_1 β_2 ... β_n
```

## Example

### Input

```
4 2
1 2
2 3
3 4
4 5
0 0 1 1
```

### Output

```
0.123456 0.789012
```

## How to Run

### 1. Build the Program

```bash
cd hard/logistic_regression
make
```

Or build from the top-level hard directory:

```bash
cd hard
make logistic_regression
```

To compile for a specific GPU architecture:

```bash
make GPU_ARCH=gfx90a    # For AMD MI210
make GPU_ARCH=gfx908    # For AMD MI100
make GPU_ARCH=gfx1100   # For AMD Radeon W7900
```

This builds `main` executable from the combined `main.hip` source.

**Optional: Build legacy versions**

```bash
make all_versions  # Builds main, exe_main, and exe_fs_main
```

### 2. Generate Test Cases (Optional)

```bash
python3 geninput.py
```

### 3. Run the Program

The combined `main` executable supports both stdin and file input:

**Option A: Interactive input (stdin)**

```bash
./main
```

Then enter the input manually.

**Option B: File input**

```bash
./main testcases/1.in
```

**Option C: Pipe input**

```bash
echo "4 2
1 2
2 3
3 4
4 5
0 0 1 1" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
