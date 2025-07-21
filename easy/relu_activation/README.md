# ReLU Activation 

## Description
Implement a program that applies the ReLU (Rectified Linear Unit) activation function to an array of 32-bit floating point numbers on a GPU. The program should take an input array and produce an output array where each element is the result of applying ReLU function: f(x) = max(0, x).

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
Output N floating point numbers representing the ReLU activation result, separated by spaces, with a newline at the end.

Output format:
```bash
relu(a1) relu(a2) ... relu(aN)
```

Where relu(x) = max(0, x) for each element x in the input array.

## Example

### Input
```
6
-2.5 1.3 0.0 -5.7 3.8 -1.0
```

### Output
```
0 1.3 0 0 3.8 0
```