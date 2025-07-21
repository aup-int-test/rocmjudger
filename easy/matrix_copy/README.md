# Matrix Copy

## Description
Implement a program that copies a square matrix containing 32-bit floating point numbers on a GPU. The program should take an input matrix A (N×N) and copy it to matrix B using GPU operations.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The operation should be performed on the GPU

## Input Description
You will be given 1 value N, followed by N×N floating point values for the matrix.

Input format:
```bash
N
a11 a12 ... a1N
a21 a22 ... a2N
...
aN1 aN2 ... aNN
```

Constraints:
- 1 ≤ N ≤ 10000, Matrix dimension(integer)
- aij, Matrix values(float)

## Output Description
Output N×N floating point numbers representing the original matrix A, with each row on a new line and values separated by spaces.

Output format:
```bash
a11 a12 ... a1N
a21 a22 ... a2N
...
aN1 aN2 ... aNN
```

Where the output is identical to the input matrix.

## Example

### Input
```
2
1.5 2.3
4.1 5.9
```

### Output
```
1.5 2.3
4.1 5.9
```