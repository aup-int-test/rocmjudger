# Softmax

## Description

Implement a GPU program that applies the **numerically stable Softmax activation function** to an array of 32-bit floating-point numbers.

For an input vector $x = [x_1, x_2, \dots, x_n]$, the output is

$$
\sigma(x)_i = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$

where $m = \max(x)$.
The subtraction of the maximum value $m$ ensures numerical stability.

* **No external libraries** are allowed.
* The `solve()` function signature must remain unchanged.
* The final results must be stored in the output array.

## Input

* First line: integer `N` — the length of the array (1 ≤ N ≤ 100000000).
* Second line: `N` space-separated 32-bit floating-point numbers.

Example:

```
3
1.0 2.0 3.0
```

## Output

* `N` space-separated floating-point numbers representing the Softmax values, printed with a trailing newline.
* Each value is

  $$
  \operatorname{softmax}(a_i) = \frac{\exp(a_i - \max(a))}{\sum_j \exp(a_j - \max(a))}
  $$

  and the outputs sum to **1.0**.

Example:

```
0.090031 0.244728 0.665240
```