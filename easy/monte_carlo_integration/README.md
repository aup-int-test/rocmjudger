# Monte Carlo Integration

## Description
Implement a program that performs Monte Carlo numerical integration on a GPU using shared memory optimization. The program should take an array of y-values and integration bounds, then compute the approximate integral using parallel reduction with shared memory.

- External libraries are not permitted
- The solve function signature must remain unchanged
- Must use shared memory and parallel reduction for optimization
- The final result must be stored in the result variable

## Input Description
You will be given 3 values: a, b, and n_samples, followed by n_samples double values representing y-coordinates.

Input format:
```bash
a b n_samples
y1 y2 ... y_n_samples
```

Constraints:
- -1000000 ≤ a, b ≤ 1000000, Integration bounds(double)
- 1 ≤ n_samples ≤ 100000000, Number of samples(integer)
- yi, Sample y-values(double)

## Output Description
Output a single double value representing the Monte Carlo integration result.

Output format:
```bash
result
```

Where result approximates the integral using the formula: (b-a) * (sum of y_samples) / n_samples

## Example

### Input
```
0 2 8
0.0625 0.25 0.5625 1.0 1.5625 2.25 3.0625 4.0
```

### Output
```
3.125000
```