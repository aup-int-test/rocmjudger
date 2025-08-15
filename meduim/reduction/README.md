# Array Reduction

## Description
Implement a program that performs parallel reduction (sum) of an array of 32-bit floating point numbers on a GPU. The program should take an input array and compute the sum of all elements using GPU parallel reduction techniques.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in the output variable

## Input Description
You will be given 1 value N, followed by N floating point values.

Input format:
```bash
N
a1 a2 ... aN
```

Constraints:
- 1 ≤ N ≤ 1000000000, Length of array(integer)
- ai, Array values(integer)

## Output Description
Output a single integer number representing the sum of all array elements.

Output format:
```bash
sum
```

Where sum = a1 + a2 + ... + aN

## Example

### Input
```
5
1 2 3 4 5
```

### Output
```
15
```