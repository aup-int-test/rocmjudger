# Monte Carlo Integration

## Description
Implement a program that performs **Monte Carlo numerical integration** on a GPU using shared memory optimization.  
The program should take an array of y-values and integration bounds, then compute the approximate integral using **parallel reduction with shared memory**.

### Algorithm
The Monte Carlo estimation of the definite integral is:

$$
\int_a^b f(x)\,dx \approx (b - a) \cdot \frac{1}{n} \sum_{i=1}^{n} y_i
$$

Where:
- \(a\) and \(b\) are the integration bounds
- \(n\) is the number of samples
- \(y_i = f(x_i)\) are the sampled function values

**Steps**  
1. Sum all y-values \(\sum_{i=1}^{n} y_i\) using GPU parallel reduction with shared memory.  
2. Multiply the average \(\frac{1}{n}\sum y_i\) by \((b - a)\) to approximate the integral.

**Requirements**
- External libraries are **not permitted**  
- The `solve` function signature must remain unchanged  
- The final result must be stored in the `result` variable  

## Input
You will be given 3 values: `a`, `b`, and `n_samples`, followed by `n_samples` double values representing y-coordinates.

**Format**
```

a b n_samples
y1 y2 ... y_n_samples

```

Constraints:
- -1,000,000 ≤ a, b ≤ 1,000,000 (double)
- 1 ≤ n_samples ≤ 100,000,000 (integer)
- yᵢ: Sample y-values (double)

## Output
Output a single double value representing the Monte Carlo integration result.

**Format**
```

result

```

Where  
\[
\text{result} \approx (b - a) \times \frac{1}{n} \sum_{i=1}^{n} y_i
\]

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