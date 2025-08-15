# 1D Convolution

## Description
Implement a program that performs 1D convolution of an input signal with a kernel on a GPU. The program should take an input vector and a kernel vector, then produce an output vector containing the convolution result.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in the output array

## Input Description
You will be given 2 values: input_size and kernel_size, followed by input_size input values and kernel_size kernel values.

Input format:
```bash
input_size kernel_size
a1, a2, ... a_input_size
k1, k2, ... k_kernel_size
```

Constraints:
- 1 ≤ input_size ≤ 1000000, Length of input array(integer)
- 1 ≤ kernel_size ≤ input_size, Length of kernel array(integer)
- a1, a2, ... a_input_size, Input array values(float)
- k1, k2, ... k_kernel_size, Kernel array values(float)

## Output Description
Output (input_size - kernel_size + 1) floating point numbers representing the 1D convolution result, separated by spaces, with a newline at the end.

Output format:
```bash
c1 c2 ... c_output_size
```

Where ci = Σ(j=0 to kernel_size-1) a_(i+j) * k_j for i = 0, 1, ..., output_size-1

## Example
### Input
```
5 3
1.0 2.0 3.0 4.0 5.0
0.5 1.0 0.5
```

### Output
```
4 6 8
```