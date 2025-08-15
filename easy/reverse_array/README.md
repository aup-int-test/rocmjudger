# Array Reverse

## Description
Implement a program that reverses an array of 32-bit floating point numbers on a GPU. The program should take an input array and reverse the order of its elements in-place.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in the input array

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
Output N floating point numbers representing the reversed array, separated by spaces, with a newline at the end.

Output format:
```bash
aN aN-1 ... a1
```

Where the output array contains the same elements as the input array but in reverse order.

## Example

### Input
```
5
1.0 2.0 3.0 4.0 5.0
```

### Output
```
5.0 4.0 3.0 2.0 1.0
```