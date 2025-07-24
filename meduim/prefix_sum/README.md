# Prefix Sum (Parallel Scan)

## Description
Implement a program that computes the prefix sum (parallel scan) of an array of integers on a GPU using hierarchical scanning. The program should take an input array and produce an output array where each element is the sum of all previous elements including itself.

- External libraries are not permitted
- The solve function signature must remain unchanged
- Must use hierarchical parallel scanning for efficiency
- The final result must be stored in the output array

## Input Description
You will be given 1 value N, followed by N integer values.

Input format:
```bash
N
a1 a2 ... aN
```

Constraints:
- 1 ≤ N ≤ 100000000, Length of array(integer)
- -1000 ≤ ai ≤ 1000, Array values(integer)

## Output Description
Output N floating point numbers representing the prefix sum result, separated by spaces, with a newline at the end.

Output format:
```bash
prefix_sum(a1) prefix_sum(a2) ... prefix_sum(aN)
```

Where prefix_sum(ai) = a1 + a2 + ... + ai for each position i.

## Example

### Input
```
5
1 2 3 4 5
```

### Output
```
1 3 6 10 15
```