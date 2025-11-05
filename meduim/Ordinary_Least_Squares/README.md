# Ordinary Least Squares (OLS)

## Description

Implement a GPU program to solve the **Ordinary Least Squares (OLS)** regression problem.
Given a feature matrix **X** of size **n_samples × n_features** and a target vector **y** of size **n_samples**,
compute the coefficient vector **β** that minimizes the sum of squared residuals:

[
\min_{\beta} |X\beta - y|^2
]

The closed-form solution is:

[
\beta = (X^T X)^{-1} X^T y
]

All matrices are stored in **row-major order** and use **32-bit floating point (FP32)** arithmetic.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output vector `beta`

## Input Description

The input consists of two integers `n_samples` and `n_features`,
followed by `n_samples × n_features` floating point values representing **X**,
and `n_samples` floating point values representing **y**.

Input format:

```bash
n_samples n_features
X_1 X_2 ... X_{n_samples×n_features}
y_1 y_2 ... y_{n_samples}
```

Constraints:

* 1 ≤ n_samples ≤ 100,000
* 1 ≤ n_features ≤ 1,000
* n_samples ≥ n_features
* −1000.0 ≤ Xᵢⱼ, yᵢ ≤ 1000.0

Solutions are tested with:

* Absolute tolerance ≤ 1e−2
* Relative tolerance ≤ 1e−2

## Output Description

Output `n_features` floating point numbers representing the OLS coefficient vector **β**, separated by spaces and ending with a newline.

Output format:

```bash
β1 β2 ... βn_features
```

## Example

### Input

```
3 2
1 2
2 3
3 4
1
2
3
```

### Output

```
0.000000 1.000000
```

## How to Run

### 1. Build the Program

```bash
cd hard/ordinary_least_squares
make
```

Or build from the top-level hard directory:

```bash
cd hard
make ordinary_least_squares
```

To compile for a specific GPU architecture:

```bash
make GPU_ARCH=gfx90a    # For AMD MI210
make GPU_ARCH=gfx908    # For AMD MI100
make GPU_ARCH=gfx1100   # For AMD Radeon W7900
```

This builds the `main` executable from the combined `main.hip` source.

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
echo "3 2
1 2
2 3
3 4
1
2
3" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
