# 2D Convolution

## Description
Implement a program that performs 2D convolution of an input matrix with a kernel matrix on a GPU. The program should take an input matrix and a kernel matrix, then produce an output matrix containing the convolution result.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in the output array

## Input Description
You will be given 4 values: input_rows, input_cols, kernel_rows, kernel_cols, followed by the input matrix and kernel matrix.

Input format:
```bash
input_rows input_cols kernel_rows kernel_cols
a11 a12 ... a1_input_cols
a21 a22 ... a2_input_cols
...
a_input_rows1 a_input_rows2 ... a_input_rows_input_cols
k11 k12 ... k1_kernel_cols
k21 k22 ... k2_kernel_cols
...
k_kernel_rows1 k_kernel_rows2 ... k_kernel_rows_kernel_cols
```

Constraints:
- 1 ≤ input_rows, input_cols ≤ 3072, Input matrix dimensions(integer)
- 1 ≤ kernel_rows, kernel_cols ≤ 64, Kernel matrix dimensions(integer)
- kernel_rows ≤ input_rows, kernel_cols ≤ input_cols
- aij, kij, Matrix values(float)

## Output Description
Output (input_rows - kernel_rows + 1) × (input_cols - kernel_cols + 1) floating point numbers representing the 2D convolution result, with each row on a new line and values separated by spaces.

Output format:
```bash
c11 c12 ... c1_output_cols
c21 c22 ... c2_output_cols
...
c_output_rows1 c_output_rows2 ... c_output_rows_output_cols
```

Where cij = Σ(m=0 to kernel_rows-1) Σ(n=0 to kernel_cols-1) a_(i+m)(j+n) * k_mn

## Example

### Input
```
2 3 2 3
1. 2. 3.
4. 5. 6.
1. 0.
```

### Output
```
32.
```