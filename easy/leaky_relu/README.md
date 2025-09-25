# Leaky ReLU

## Description

Implement a program that applies the **Leaky ReLU activation function** to an array of 32-bit floating point numbers on a GPU.
The program should take an input array and produce an output array where each element is transformed by the Leaky ReLU function.

**Algorithm**
The Leaky ReLU function is defined as:

$$
f(x) =
\begin{cases}
x, & x > 0 \\
\alpha x, & x \le 0
\end{cases}
$$

where $\alpha = 0.01$.

For every element $x$ in the input array, the output is computed by:

* returning $x$ itself when $x > 0$
* returning $0.01 \times x$ when $x \le 0$

**Requirements**

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array

## Input Description

You will be given 1 value `N`, followed by `N` floating point values.

Input format:

```bash
N
a1 a2 ... aN
```

Constraints:

* 1 ≤ N ≤ 10,000,000
* ai: Array values (float)

## Output Description

Output N floating point numbers representing the Leaky ReLU activation result, separated by spaces, with a newline at the end.

Output format:

```bash
leaky_relu(a1) leaky_relu(a2) ... leaky_relu(aN)
```

Where

$$
\text{leaky\_relu}(x) =
\begin{cases}
x, & x > 0 \\
0.01x, & x \le 0
\end{cases}
$$

## Example

### Input

```
6
-2.5 1.3 0.0 -5.7 3.8 -1.0
```

### Output

```
-0.025 1.3 0 -0.057 3.8 -0.01
```

---