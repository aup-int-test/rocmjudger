# Softmax 

## Description
Implement a program that applies the Softmax activation function to an array of 32-bit floating point numbers on a GPU. The program should take an input array and produce an output array where each element is the result of applying the numerically stable Softmax function: softmax(xi) = exp(xi - max) / Σ(exp(xj - max)).

- External libraries are not permitted
- The solve function signature must remain unchanged
- Must implement numerically stable version using **max subtraction**
- The final result must be stored in the output array

## Input Description
You will be given 1 value N, followed by N floating point values.

Input format:
```bash
N
a1 a2 ... aN
```

Constraints:
- 1 ≤ N ≤ 100000000, Length of array(integer)
- ai, Array values(float)

## Output Description
Output N floating point numbers representing the Softmax activation result, separated by spaces, with a newline at the end.

Output format:
```bash
softmax(a1) softmax(a2) ... softmax(aN)
```

Where softmax(ai) = exp(ai - max(a)) / Σ(exp(aj - max(a))) and the sum of all output values equals 1.0.

## Example

### Input
```
3
1.0 2.0 3.0
```

### Output
```
0.090031 0.244728 0.665240
```