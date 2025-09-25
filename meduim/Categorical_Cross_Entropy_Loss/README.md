# Categorical Cross Entropy Loss

## Description

Implement a GPU program to calculate the **categorical cross-entropy loss** for a batch of predictions.
Given a matrix of predicted **logits** $Z \in \mathbb{R}^{N \times C}$ and a vector of **true class labels** $\mathrm{true\_labels} \in \{0,\ldots,C-1\}^N$, compute the **average** cross-entropy loss over the batch.

For a single sample $j$ with logits $z_j = [z_{j1}, \ldots, z_{jC}]$ and true label $y_j$, use the numerically stable form:

$$
\mathrm{Loss}_j = \log \left( \sum_{k=1}^{C} e^{z_{jk}} \right) - z_{j y_j}
$$

The final output is the batch average:

$$
L = \frac{1}{N} \sum_{j=1}^{N} \mathrm{Loss}_j
$$

**Implementation Requirements**

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* Use a numerically stable **log-sum-exp** (e.g., subtract the row max before exponentiation)
* The final result (average loss) must be stored in the `loss` variable (a single float)

## Input Description

You will be given integers **N** (number of samples) and **C** (number of classes), followed by the $N\times C$ matrix of logits (row-major: one row per sample), and then **N** integers for the true labels.

**Input format:**

```bash
N C
z11 z12 ... z1C
z21 z22 ... z2C
...
zN1 zN2 ... zNC
y1 y2 ... yN
```

## Output Description

Output a **single floating-point number**: the average loss $L$, followed by a newline.

**Output format:**

```bash
L
```

## Example 1

### Input

```
2 3
1.0 2.0 0.5
0.1 3.0 1.5
1 1
```

### Output

```
0.354893
```

## Example 2

### Input

```
3 4
-0.5  1.5  0.0  1.0
 2.0 -1.0  0.5  0.5
 0.0  0.0  0.0  0.0
3 0 1
```

### Output

```
0.988204
```

## Constraints

* $1 \le N \le 10{,}000$
* $2 \le C \le 1{,}000$
* $-10.0 \le \mathtt{logits}[i,j] \le 10.0$
* $0 \le \mathtt{truelabels}[i] < C$


