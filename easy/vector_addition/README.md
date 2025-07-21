# Vector Addition

## Description
Implement a program that performs element-wise addition of two vectors containing 32-bit floating point numbers on a GPU. The program should take two input vectors of equal length and produce a single output vector containing their sum.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in vector C

## Input Description
You will be given 1 value N, followed by N function values.

Input format:
```bash
N  
a1, a2, ... an  
b1, b2, ... bn
```

Constraints:
- 1 ≤ N ≤ 1000000, Length of array(integer)
- a1, a2, ... an, Array values(float)
- b1, b2, ... bn, Array values(float)

## Output Description
Output N floating point numbers representing the element-wise sum of the two input vectors, formatted to 3 decimal places and separated by spaces, with a newline at the end.

Output format:
```bash
c1 c2 ... cn
```

Where ci = ai + bi for i = 1, 2, ..., n

## Example
### Input
```
5
1.5 2.3 3.7 4.1 5.9
0.8 1.2 2.3 3.4 4.6
```

### Output
```
2.300 3.500 6.000 7.500 10.500
```
