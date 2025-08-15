# Histogram

## Description
Implement a program that computes the histogram of an array of integers on a GPU using parallel computing techniques. The program should take an input array and produce an output histogram where each bin represents the count of elements with that specific value.

- External libraries are not permitted
- The solve function signature must remain unchanged
- Must use GPU parallel computing for efficiency
- The final result must be stored in the histogram array

## Input Description
You will be given 2 values N and num_bins, followed by N integer values.

Input format:
```bash
N num_bins
a1 a2 ... aN
```

Constraints:
- 1 ≤ N ≤ 100,000,000, Length of array (integer)
- 1 ≤ num_bins ≤ 1024, Number of histogram bins (integer)
- 0 ≤ ai < num_bins, Array values (integer)

## Output Description
Output num_bins integer numbers representing the histogram result, separated by spaces, with a newline at the end.

Output format:
```bash
count(0) count(1) ... count(num_bins-1)
```

Where count(i) represents the number of elements in the input array that have value i.

## Example

### Input
```
8 4
0 1 2 3 0 1 2 0
```

### Output
```
3 2 2 1
```
