# Categorical Cross Entropy Loss

## Description

Implement a GPU program that computes the **categorical cross-entropy loss** for a batch of predictions.
Given a matrix of **logits** ( Z \in \mathbb{R}^{N \times C} ) and a vector of **true class labels**
( \mathrm{true_labels} \in {0, \ldots, C-1}^N ), compute the **average** cross-entropy loss over the batch.

For each sample ( j ) with logits ( z_j = [z_{j1}, \ldots, z_{jC}] ) and true label ( y_j ), the loss is:

[
\mathrm{Loss}*j = \log \left( \sum*{k=1}^{C} e^{z_{jk}} \right) - z_{j y_j}
]

The final result is the mean loss across all samples:

[
L = \frac{1}{N} \sum_{j=1}^{N} \mathrm{Loss}_j
]

To ensure numerical stability, use **log-sum-exp** by subtracting the row-wise maximum before exponentiation.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the variable `loss` (a single float)

## Input Description

You will be given integers `N` (number of samples) and `C` (number of classes), followed by `N×C` floating-point numbers (the logits), and `N` integers (the true class labels).

Input format:

```bash
N C
z11 z12 ... z1C
z21 z22 ... z2C
...
zN1 zN2 ... zNC
y1 y2 ... yN
```

Constraints:

* 1 ≤ N ≤ 10,000
* 2 ≤ C ≤ 1,000
* −10.0 ≤ logits[i,j] ≤ 10.0
* 0 ≤ true_labels[i] < C

## Output Description

Output a single floating-point number representing the **average categorical cross-entropy loss**:

Output format:

```bash
L
```

## Example 1

### Input

```
2 3
1.0 2.0 0.5
0.1 3.0 1.5
1 1
```

### Output

```
0.354893
```

## Example 2

### Input

```
3 4
-0.5  1.5  0.0  1.0
 2.0 -1.0  0.5  0.5
 0.0  0.0  0.0  0.0
3 0 1
```

### Output

```
0.988204
```

## How to Run

### 1. Build the Program

```bash
cd medium/categorical_cross_entropy
make
```

Or build from the top-level medium directory:

```bash
cd medium
make categorical_cross_entropy
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
echo "2 3
1.0 2.0 0.5
0.1 3.0 1.5
1 1" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
