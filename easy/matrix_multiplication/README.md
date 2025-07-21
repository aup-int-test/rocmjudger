# Matrix Multiplication

## Description
Implement a program that performs matrix multiplication of two matrices containing 32-bit floating point numbers on a GPU. The program should take two input matrices A (M×N) and B (N×K) and produce a single output matrix C (M×K) containing their product.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in matrix C

## Input Description
You will be given 3 values M, N, K, followed by M×N values for matrix A and N×K values for matrix B.

Input format:
```bash
M N K
a11 a12 ... a1N
a21 a22 ... a2N
...
aM1 aM2 ... aMN
b11 b12 ... b1K
b21 b22 ... b2K
...
bN1 bN2 ... bNK
```

Constraints:
- 1 ≤ M, N, K ≤ 1000, Matrix dimensions(integer)
- aij, bij, Matrix values(float)

## Output Description
Output M×K floating point numbers representing the matrix multiplication result, formatted to 3 decimal places and separated by spaces, with each row on a new line.

Output format:
```bash
c11 c12 ... c1K
c21 c22 ... c2K
...
cM1 cM2 ... cMK
```

Where cij = Σ(k=1 to N) aik * bkj for i = 1, 2, ..., M and j = 1, 2, ..., K

## Example

### Input
```
2 3 2
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0
9.0 10.0
11.0 12.0
```

### Output
```
58.000  64.000
139.000 154.000
```