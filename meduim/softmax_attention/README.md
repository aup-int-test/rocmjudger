# Softmax Attention

## Description

Implement a GPU program that computes **scaled dot-product attention** using Query (Q), Key (K), and Value (V) matrices.
For $Q\in\mathbb{R}^{M\times d}$, $K\in\mathbb{R}^{N\times d}$, $V\in\mathbb{R}^{N\times d}$, the output $O\in\mathbb{R}^{M\times d}$ is

$$
\mathrm{Attention}(Q,K,V) \;=\; \mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d}}\right)\, V 
$$

* **Numerical stability:** apply row-wise softmax with **max subtraction**. For logits $L = QK^{\mathsf T}/\sqrt{d}$ and each row $i$,

  $$
  s_{ij}=\frac{\exp(L_{ij}-\max_j L_{ij})}{\sum_{k=1}^{N}\exp(L_{ik}-\max_j L_{ij})}
  $$
* **No external libraries** are allowed.
* The `solve()` function signature must remain unchanged.
* The final results must be stored in the output array.

## Input

You will be given integers `M N d`, followed by matrices **Q (M×d)**, **K (N×d)**, and **V (N×d)** in row-major order.

Format:

```
M N d
q11 q12 ... q1d
...
qM1 qM2 ... qMd
k11 k12 ... k1d
...
kN1 kN2 ... kNd
v11 v12 ... v1d
...
vN1 vN2 ... vNd
```

**Constraints**

* $1 \le M,N \le 4096$
* $1 \le d \le 512$
* All values are 32-bit floats.

## Output

Print the attention output matrix $O$ of shape $M\times d$.
Each row on a new line; values are space-separated with a trailing newline.

Format:

```
o11 o12 ... o1d
o21 o22 ... o2d
...
oM1 oM2 ... oMd
```

## Notes (implementation hints)

* Compute logits $L = QK^{\mathsf T}/\sqrt{d}$ using GPU parallelism (e.g., tiled matmul).
* For each row of $L$: reduce to find `max`, compute `exp(L - max)`, reduce to get the row sum, then normalize.
* Multiply the row-wise attention weights with $V$ to produce $O$.
* The outputs in each row of the softmax weights sum to **1.0**.

## Example

**Input**

```
2 3 4
1. 0. 0. 0.
0. 1. 0. 0.
1. 0. 0. 0.
0. 1. 0. 0.
0. 0. 1. 0.
1. 2. 3. 4.
5. 6. 7. 8.
9. 10. 11. 12.
```

**Output**

```
4.29 5.29 6.29 7.29
5.00 6.00 7.00 8.00
```