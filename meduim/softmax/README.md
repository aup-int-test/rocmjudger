# Softmax

## Description

Implement a GPU program that applies the **numerically stable Softmax activation function** to an array of 32-bit floating-point numbers.
For an input vector ( x = [x_1, x_2, \dots, x_n] ), the output is:

[
\sigma(x)*i = \frac{e^{x_i - m}}{\sum*{j=1}^{n} e^{x_j - m}}
]
where ( m = \max(x) ) for numerical stability.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array

## Input Description

You will be given one integer `N`, followed by `N` floating-point numbers.

Input format:

```bash
N
a1 a2 ... aN
```

Constraints:

* 1 ≤ N ≤ 100,000,000
* aᵢ: Array values (float)

## Output Description

Output `N` floating-point numbers representing the Softmax results, separated by spaces, with a newline at the end.
Each value satisfies:

[
\operatorname{softmax}(a_i) = \frac{\exp(a_i - \max(a))}{\sum_j \exp(a_j - \max(a))}
]

and all outputs sum to **1.0**.

Output format:

```bash
s1 s2 ... sN
```

## Example

### Input

```
3
1.0 2.0 3.0
```

### Output

```
0.090031 0.244728 0.665240
```

## How to Run

### 1. Build the Program

```bash
cd easy/softmax
make
```

Or build from the top-level easy directory:

```bash
cd easy
make softmax
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
echo "3
1.0 2.0 3.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
