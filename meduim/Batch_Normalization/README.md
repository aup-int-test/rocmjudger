# Batch Normalization

## Description

Implement **Batch Normalization** in **FP32**.
Given an input matrix **X** of shape `[N, C]`, along with scale parameters **γ (gamma)** and bias parameters **β (beta)**, compute the normalized output matrix **Y** of the same shape.

For each channel **c**, normalization is defined as:

[
\mu_c = \frac{1}{N} \sum_{i=1}^{N} X_{i,c}
]

[
\sigma_c^2 = \frac{1}{N} \sum_{i=1}^{N} (X_{i,c} - \mu_c)^2
]

[
Y_{i,c} = \gamma_c \cdot \frac{X_{i,c} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c
]

All computations must use **32-bit floating point numbers (FP32)**.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the **output matrix Y**

## Input Description

You will be given integers `N` and `C`, a floating-point number `ε (epsilon)`, followed by the matrix **X**, and parameters **γ** and **β**.

Input format:

```bash
N C ε
X_1 X_2 ... X_{N×C}
γ_1 γ_2 ... γ_C
β_1 β_2 ... β_C
```

Constraints:

* 1 ≤ N ≤ 10,000
* 1 ≤ C ≤ 1,024
* All values are 32-bit floating point numbers

## Output Description

Output `N × C` floating-point numbers representing the normalized result matrix **Y**, printed in row-major order.
Each row corresponds to one sample, and each value should be space-separated.

Output format:

```bash
Y_1_1 Y_1_2 ... Y_1_C
...
Y_N_1 Y_N_2 ... Y_N_C
```

## Example

### Input

```
3 2 1e-5
1.0 2.0
3.0 4.0
5.0 6.0
1.0 1.0
0.0 0.0
```

### Output

```
-1.224744 -1.224744
0.000000 0.000000
1.224744 1.224744
```

## How to Run

### 1. Build the Program

```bash
cd medium/batch_normalization
make
```

Or build from the top-level medium directory:

```bash
cd medium
make batch_normalization
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
echo "3 2 1e-5
1.0 2.0
3.0 4.0
5.0 6.0
1.0 1.0
0.0 0.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
