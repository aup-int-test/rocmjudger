# Batched Matrix Multiplication

## Description

Implement a GPU program that performs **batched matrix multiplication** in **FP32**.
Given a batch of matrices **A** with shape `[B, M, K]` and a batch of matrices **B** with shape `[B, K, N]`, compute the output batch **C** with shape `[B, M, N]`, such that for each batch index **b**:

[
C_b = A_b \times B_b
]

All matrices are stored in **row-major order** and use **32-bit floating point numbers (FP32)**.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the **output array C**

## Input Description

You will be given integers `B`, `M`, `N`, and `K`, followed by all elements of **A** and **B** in row-major order.

Input format:

```bash
B M N K
A_1 A_2 ... A_{B×M×K}
B_1 B_2 ... B_{B×K×N}
```

Constraints:

* 1 ≤ B ≤ 128
* 1 ≤ M, N, K ≤ 1024
* All values are 32-bit floats (FP32)

## Output Description

Output `B × M × N` floating point numbers representing the result matrices **C**, printed in row-major order.
Each matrix in the batch should be separated by a newline after its last row.

Output format:

```bash
C_1_1 C_1_2 ... C_1_N
...
C_M_1 C_M_2 ... C_M_N
```

(repeated for all batches)

## Example

### Input

```
2 2 2 3
1 2 3
4 5 6
7 8 9
10 11 12
1 2
3 4
5 6
6 5
4 3
2 1
```

### Output

```
22 28
49 64
92 68
128 95
```

## How to Run

### 1. Build the Program

```bash
cd medium/batched_matrix_multiplication
make
```

Or build from the top-level medium directory:

```bash
cd medium
make batched_matrix_multiplication
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
echo "2 2 2 3
1 2 3
4 5 6
7 8 9
10 11 12
1 2
3 4
5 6
6 5
4 3
2 1" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
