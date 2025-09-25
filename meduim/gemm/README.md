# General Matrix Multiplication (GEMM)

## Description

Implement a basic General Matrix Multiplication (GEMM). Given matrix **A** of dimensions **M × K**, matrix **B** of dimensions **K × N**, input/output matrix **C** of dimensions **M × N**, and scalar multipliers **α** and **β**, compute:

$$
C = \alpha \cdot (A \times B) + \beta \cdot C_{\text{initial}}
$$

* All input matrices **A**, **B**, and the initial state **C_initial** contain 16-bit floating-point numbers (FP16 / `half`) in **row-major** order.
* **α** and **β** are 32-bit floats (FP32).
* Accumulation during multiplication must use FP32 for better precision before converting the final result to FP16.

**Implementation Requirements**

* Use only native features (external libraries other than WMMA are not permitted).
* The `solve` function signature must remain unchanged.
* Accumulate in FP32, then convert to FP16 for the final store.
* The final result must be written back into matrix **C** as `half`.

## Input Description

You will be given **M**, **N**, **K**, followed by matrices **A**, **B**, **C_initial** (all in row-major FP16), and then scalars **α** and **β** (FP32).

**Input format:**

```bash
M N K
A11 A12 ... A1K
A21 A22 ... A2K
...
AM1 AM2 ... AMK
B11 B12 ... B1N
B21 B22 ... B2N
...
BK1 BK2 ... BKN
C11 C12 ... C1N
C21 C22 ... C2N
...
CM1 CM2 ... CMN
alpha beta
```

**Constraints**

* 16 ≤ **M**, **N**, **K** ≤ 4096
* Matrix elements are FP16; α, β are FP32

## Output Description

Output matrix **C** (**M × N**, FP16, row-major). Print **M** lines, each with **N** values separated by spaces, followed by a newline.

**Output format:**

```bash
C11 C12 ... C1N
C21 C22 ... C2N
...
CM1 CM2 ... CMN
```

## Example

### Input

```
2 2 3
1.0 2.0 3.0
4.0 5.0 6.0
1.0 2.0
3.0 4.0
5.0 6.0
1.0 1.0
1.0 1.0
1.0 0.0
```

### Explanation

* A (M=2, K=3):

  ```
  [1.0 2.0 3.0]
  [4.0 5.0 6.0]
  ```
* B (K=3, N=2):

  ```
  [1.0 2.0]
  [3.0 4.0]
  [5.0 6.0]
  ```
* C_initial (M=2, N=2):

  ```
  [1.0 1.0]
  [1.0 1.0]
  ```
* α = 1.0 (FP32), β = 0.0 (FP32)

### Output

```
22.0 28.0
49.0 64.0
```
