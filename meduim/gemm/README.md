# General Matrix Multiplication (GEMM)

## Description

Implement a basic **General Matrix Multiplication (GEMM)** on a GPU.
Given matrix **A (M×K)**, matrix **B (K×N)**, input/output matrix **C (M×N)**, and scalar multipliers **α** and **β**, compute:

[
C = \alpha \cdot (A \times B) + \beta \cdot C_{\text{initial}}
]

* All input matrices **A**, **B**, and the initial state **C_initial** contain **16-bit floating-point numbers (FP16 / `half`)** in **row-major** order.
* **α** and **β** are **32-bit floats (FP32)**.
* Accumulation during multiplication must use **FP32** before converting the final result to FP16.
* **External libraries other than WMMA are not permitted**
* The `solve` function signature must remain unchanged
* Accumulate in FP32, then convert to FP16 for the final store
* The final result must be written back into matrix **C** as `half`

## Input Description

You will be given **M**, **N**, **K**, followed by matrices **A**, **B**, **C_initial** (all in row-major FP16), and then scalars **α** and **β** (FP32).

Input format:

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

Constraints:

* 16 ≤ M, N, K ≤ 4096
* Matrix elements are FP16; α, β are FP32

## Output Description

Output matrix **C** (**M × N**, FP16, row-major). Print **M** lines, each with **N** values separated by spaces, followed by a newline.

Output format:

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

### Output

```
22.0 28.0
49.0 64.0
```

## How to Run

### 1. Build the Program

```bash
cd hard/gemm
make
```

Or build from the top-level hard directory:

```bash
cd hard
make gemm
```

To compile for a specific GPU architecture:

```bash
make GPU_ARCH=gfx90a    # For AMD MI210
make GPU_ARCH=gfx908    # For AMD MI100
make GPU_ARCH=gfx1100   # For AMD Radeon W7900
```

This builds `main` executable from the combined `main.hip` source.

**Optional: Build legacy versions**

```bash
make all_versions  # Builds main, exe_main, and exe_fs_main
```

### 2. Generate Test Cases (Optional)

```bash
python3 geninput.py
```

### 3. Run the Program

The combined `main` executable supports both stdin and file input:

**Option A: Interactive input (stdin)**

```bash
./main
```

Then enter the input manually.

**Option B: File input**

```bash
./main testcases/1.in
```

**Option C: Pipe input**

```bash
echo "2 2 3
1.0 2.0 3.0
4.0 5.0 6.0
1.0 2.0
3.0 4.0
5.0 6.0
1.0 1.0
1.0 1.0
1.0 0.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
