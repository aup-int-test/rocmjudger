# Mean Squared Error (MSE)

## Description

Implement a GPU program to compute the **Mean Squared Error (MSE)** between predicted values and target values.
Given two arrays of equal length, **predictions** and **targets**, the loss is defined as

$$
\mathrm{MSE} \;=\; \frac{1}{N}\sum_{i=1}^{N}\bigl(\text{predictions}_i - \text{targets}_i\bigr)^2
$$

where **N** is the number of elements.

* **No external libraries** are permitted.
* The `solve()` function signature must remain unchanged.
* The final result must be stored in the variable `mse`.

## Input

* First line: integer `N` — the number of elements (1 ≤ N ≤ 100,000,000).
* Second line: `N` floating-point numbers representing the **predictions** array.
* Third line: `N` floating-point numbers representing the **targets** array.

Example:

```
4
1.0 2.0 3.0 4.0
1.5 2.5 3.5 4.5
```

## Output

A single floating-point number — the computed MSE — printed with a newline.
Example output for the input above:

```
0.25
```

## Constraints

* $1 \le N \le 100{,}000{,}000$
* $-1000.0 \le \text{predictions}[i], \text{targets}[i] \le 1000.0$

## Examples

### Example 1

**Input**

```
4
1.0 2.0 3.0 4.0
1.5 2.5 3.5 4.5
```

**Output**

```
0.25
```

### Example 2

**Input**

```
3
10.0 20.0 30.0
12.0 18.0 33.0
```

**Output**

```
5.67
```