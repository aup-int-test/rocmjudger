

# Matrix Power

## Description

Implement a GPU program that raises a square matrix **A** of size `N × N`
to an integer power **P** using **HIP**.

You must compute:

$$
\text{output} = A^P
$$

where matrix multiplication follows standard dense multiplication
over 32-bit floating point numbers.

The implementation should utilize **shared memory tiling** for better performance
when performing matrix multiplication on the GPU.

---

## Implementation Requirements

* **External libraries are not permitted**
* The `solve` function signature must remain unchanged
* The final result must be written to the **output** array in **row-major** order
* Use shared memory and synchronization to reduce global memory traffic

---

## Input Description

The input file provides the matrix dimension **N**,
the exponent **P**, and then the elements of matrix **A** in row-major order.

**Input format:**

```bash
N P
A_11 A_12 ... A_1N
A_21 A_22 ... A_2N
...
A_N1 A_N2 ... A_NN
```

### Example

```
3 2
1 2 3
4 5 6
7 8 9
```

---

## Output Description

Output the resulting matrix ( A^P ) in **row-major order**,
formatted as `N` lines with `N` floating-point numbers per line.

**Output format:**

```bash
O_11 O_12 ... O_1N
O_21 O_22 ... O_2N
...
O_N1 O_N2 ... O_NN
```

### Example

```
30.0000 36.0000 42.0000
66.0000 81.0000 96.0000
102.0000 126.0000 150.0000
```

---

## Constraints

```
1 ≤ N ≤ 1024
1 ≤ P ≤ 20
-10.0 ≤ A_ij ≤ 10.0
```

Solutions are tested with:

* Absolute tolerance ≤ 1e-2
* Relative tolerance ≤ 1e-2

---

## Example

**Input:**

```
3 3
1 2 3
4 5 6
7 8 9
```

**Output:**

```
468.0000 576.0000 684.0000
1062.0000 1305.0000 1548.0000
1656.0000 2034.0000 2412.0000
```
