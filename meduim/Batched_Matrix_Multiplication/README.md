# Batched Matrix Multiplication

## Description

Implement a batched matrix multiplication in **FP32**.
Given a batch of matrices **A** with shape **[B, M, K]** and a batch of matrices **B** with shape **[B, K, N]**, compute the output batch **C** with shape **[B, M, N]**, such that for each batch index **b**:

$$
C_b = A_b \times B_b
$$

All matrices are stored in **row-major order** and use **32-bit floating point numbers (FP32)**.

---

## Input Description

The input consists of:

* Four integers **B**, **M**, **N**, **K**, representing the batch size and the dimensions of matrices.
* Followed by **B × M × K** floating-point numbers representing all matrices **A**.
* Followed by **B × K × N** floating-point numbers representing all matrices **B**.

**Input format:**

```bash
B M N K
A_1 A_2 ... A_{B×M×K}
B_1 B_2 ... B_{B×K×N}
```

---

## Output Description

Output **B × M × N** floating-point numbers representing the result matrices **C**, printed in row-major order.
Each matrix in the batch should be separated by a newline after its last row.

**Output format:**

```bash
C_1_1 C_1_2 ... C_1_N
...
C_M_1 C_M_2 ... C_M_N
```

(repeat for all batches)

---

## Constraints

```
1 ≤ B ≤ 128
1 ≤ M, N, K ≤ 1024
All values are 32-bit floating point numbers.
```

---

## Example

**Input:**

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

**Output:**

```
22 28
49 64
92 68
128 95
```
