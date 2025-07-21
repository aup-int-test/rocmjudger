# Matrix Transpose

## Description
Implement a program that performs matrix transpose of a matrix containing 32-bit floating point numbers on a GPU. The program should take an input matrix A (rows×cols) and produce a single output matrix containing the transposed result where A^T[j][i] = A[i][j].

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in the output array

## Input Description
You will be given 2 values: rows and cols, followed by rows×cols values for the matrix.

Input format:
```bash
rows cols
a11 a12 ... a1_cols
a21 a22 ... a2_cols
...
a_rows1 a_rows2 ... a_rows_cols
```

Constraints:
- 1 ≤ rows, cols ≤ 3000, Matrix dimensions(integer)
- aij, Matrix values(float)

## Output Description
Output rows×cols floating point numbers representing the transposed matrix, separated by spaces, with each row on a new line.

Output format:
```bash
a11 a21 ... a_rows1
a12 a22 ... a_rows2
...
a1_cols a2_cols ... a_rows_cols
```

Where the output matrix has dimensions cols×rows, and output[j][i] = input[i][j]

## Example

### Input
```
2 3
1.0 2.0 3.0
4.0 5.0 6.0
```

### Output
```
1 4
2 5
3 6
```