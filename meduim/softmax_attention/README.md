# Softmax Attetion

## Description
Implement a program that computes the attention mechanism on a GPU using Query (Q), Key (K), and Value (V) matrices. The program should perform the scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d)V, where d is the dimension of the key vectors.

- External libraries are not permitted
- The solve function signature must remain unchanged
- Must implement numerically stable softmax
- The final result must be stored in the output array

## Input Description
You will be given 3 values: M, N, d, followed by the Q matrix (M×d), K matrix (N×d), and V matrix (N×d).

Input format:
```bash
M N d
q11 q12 ... q1d
q21 q22 ... q2d
...
qM1 qM2 ... qMd
k11 k12 ... k1d
k21 k22 ... k2d
...
kN1 kN2 ... kNd
v11 v12 ... v1d
v21 v22 ... v2d
...
vN1 vN2 ... vNd
```

Constraints:
- 1 ≤ M, N ≤ 100000, Sequence lengths(integer)
- 1 ≤ d ≤ 1024, Feature dimension(integer)
- qij, kij, vij, Matrix values(float)

## Output Description
Output M×d floating point numbers representing the attention output, with each row on a new line and values separated by spaces.

Output format:
```bash
o11 o12 ... o1d
o21 o22 ... o2d
...
oM1 oM2 ... oMd
```

Where the output is computed as: softmax(QK^T/√d)V

## Example

### Input
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

### Output
```
4.29 5.29 6.29 7.29
5. 6. 7. 8.
```