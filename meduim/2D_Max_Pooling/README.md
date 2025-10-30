# 2D Max Pooling

## Description

Implement a **2D Max Pooling** operation for image or feature map downsampling in **FP32**.
Given an input tensor **input** of shape **[N, C, H, W]**, apply max pooling with the specified **kernel size**, **stride**, and **padding**, and output the tensor **output** of shape **[N, C, H_out, W_out]**.

For each output position `(n, c, h_out, w_out)`, compute:

$$
output[n, c, h_{out}, w_{out}] =
\max_{0 \le i < kernel_size,, 0 \le j < kernel_size}
input[n, c, h_{out} \times stride + i - padding,, w_{out} \times stride + j - padding]
$$

All tensors use **row-major order** and contain **32-bit floating point** values.

---

## Input Description

The input file contains:

* Seven integers:
  **N**, **C**, **H**, **W**, **kernel_size**, **stride**, **padding**
  representing batch size, channels, input height, input width, kernel size, stride, and padding.
* Followed by **N × C × H × W** floating-point numbers representing all input tensor elements in row-major order.

**Input format:**

```bash
N C H W kernel_size stride padding
input_1 input_2 ... input_{N×C×H×W}
```

---

## Output Description

Output **N × C × H_out × W_out** floating-point numbers representing the result tensor,
where

$$
H_{out} = \frac{H + 2 \times padding - kernel_size}{stride} + 1
$$

$$
W_{out} = \frac{W + 2 \times padding - kernel_size}{stride} + 1
$$

Each channel and batch should be printed in row-major order,
with spaces between elements of the same row and a newline after each row.
Leave an empty line between channels and batches.

**Output format:**

```bash
output[n=0, c=0]
y_0_0 y_0_1 ... y_0_Wout
...
(repeat for all n, c)
```

---

## Constraints

```
1 ≤ N, C ≤ 8
2 ≤ H, W ≤ 1024
1 ≤ kernel_size ≤ min(H, W)
1 ≤ stride ≤ kernel_size
0 ≤ padding ≤ kernel_size / 2
All values are 32-bit floating point numbers.
```

---

## Example

**Input:**

```
1 1 3 3 2 1 0
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
```

**Output:**

```
5.000000 6.000000
8.000000 9.000000
```

---