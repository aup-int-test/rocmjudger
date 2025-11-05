# Softmax Attention

## Description

Implement a GPU program that computes **scaled dot-product attention** using Query (Q), Key (K), and Value (V) matrices.
For ( Q\in\mathbb{R}^{M\times d} ), ( K\in\mathbb{R}^{N\times d} ), and ( V\in\mathbb{R}^{N\times d} ), the output ( O\in\mathbb{R}^{M\times d} ) is:

[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}!\left(\frac{QK^{\mathsf T}}{\sqrt{d}}\right)V
]

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array

### Numerical Stability

Row-wise softmax must include **max subtraction**.
For logits ( L = QK^{\mathsf T}/\sqrt{d} ) and each row ( i ):

[
s_{ij} = \frac{\exp(L_{ij}-\max_j L_{ij})}{\sum_{k=1}^{N}\exp(L_{ik}-\max_j L_{ij})}
]

## Input Description

You will be given three integers `M N d`, followed by matrices `Q (M×d)`, `K (N×d)`, and `V (N×d)` in row-major order.

Input format:

```bash
M N d
q11 q12 ... q1d
...
qM1 qM2 ... qMd
k11 k12 ... k1d
...
kN1 kN2 ... kNd
v11 v12 ... v1d
...
vN1 vN2 ... vNd
```

Constraints:

* 1 ≤ M, N ≤ 4096
* 1 ≤ d ≤ 512
* All values are 32-bit floats

## Output Description

Output the attention result matrix `O (M×d)` with each row on a new line and values separated by spaces.

Output format:

```bash
o11 o12 ... o1d
o21 o22 ... o2d
...
oM1 oM2 ... oMd
```

Where
[
O = \mathrm{softmax}!\left(\frac{QK^{\mathsf T}}{\sqrt{d}}\right)V
]

## Example

### Input

```
2 3 4
1. 0. 0. 0.
0. 1. 0. 0.
1. 0. 0. 0.
0. 1. 0. 0.
0. 0. 1. 0.
1. 2. 3. 4.
5. 6. 7. 8.
9. 10. 11. 12.
```

### Output

```
4.29 5.29 6.29 7.29
5.00 6.00 7.00 8.00
```

## How to Run

### 1. Build the Program

```bash
cd hard/softmax_attention
make
```

Or build from the top-level hard directory:

```bash
cd hard
make softmax_attention
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
echo "2 3 4
1. 0. 0. 0.
0. 1. 0. 0.
1. 0. 0. 0.
0. 1. 0. 0.
0. 0. 1. 0.
1. 2. 3. 4.
5. 6. 7. 8.
9. 10. 11. 12." | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
