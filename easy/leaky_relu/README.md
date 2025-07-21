# Leaky ReLU Activation Function

## Description
Implement a program that applies the Leaky ReLU activation function to an array of 32-bit floating point numbers on a GPU. The program should take an input array and produce an output array where each element is the result of applying Leaky ReLU function: f(x) = x if x > 0, else α*x where α = 0.01.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in the output array

## Input Description
You will be given 1 value N, followed by N floating point values.

Input format:
```bash
N
a1 a2 ... aN
```

Constraints:
- 1 ≤ N ≤ 10000000, Length of array(integer)
- ai, Array values(float)

## Output Description
Output N floating point numbers representing the Leaky ReLU activation result, separated by spaces, with a newline at the end.

Output format:
```bash
leaky_relu(a1) leaky_relu(a2) ... leaky_relu(aN)
```

Where leaky_relu(x) = x if x > 0, else 0.01*x for each element x in the input array.

## Example

### Input
```
6
-2.5 1.3 0.0 -5.7 3.8 -1.0
```

### Output
```
-0.025 1.3 0 -0.057 3.8 -0.01
```