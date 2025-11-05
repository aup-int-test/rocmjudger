# Mean Squared Error (MSE)

## Description

Implement a GPU program that computes the **Mean Squared Error (MSE)** between predicted values and target values.
Given two arrays of equal length — **predictions** and **targets** — the loss is defined as:

[
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\text{predictions}_i - \text{targets}_i)^2
]

where **N** is the number of elements.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the variable `mse`

## Input Description

You will be given one integer `N`, followed by two lines of `N` floating-point values representing the prediction and target arrays.

Input format:

```bash
N
pred_1 pred_2 ... pred_N
target_1 target_2 ... target_N
```

Constraints:

* 1 ≤ N ≤ 100,000,000
* −1000.0 ≤ predictions[i], targets[i] ≤ 1000.0

## Output Description

Output a single floating-point number representing the computed Mean Squared Error, followed by a newline.

Output format:

```bash
mse
```

Where
[
\text{mse} = \frac{1}{N}\sum_{i=1}^{N}(\text{pred}_i - \text{target}_i)^2
]

## Example

### Input

```
4
1.0 2.0 3.0 4.0
1.5 2.5 3.5 4.5
```

### Output

```
0.25
```

## How to Run

### 1. Build the Program

```bash
cd easy/mean_squared_error
make
```

Or build from the top-level easy directory:

```bash
cd easy
make mean_squared_error
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
echo "4
1.0 2.0 3.0 4.0
1.5 2.5 3.5 4.5" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
