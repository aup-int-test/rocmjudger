# Matrix Power

## Description

Implement a GPU program that raises a square matrix **A (N×N)** to an integer power **P** using HIP.
Compute:
[
\text{output} = A^P
]
with standard dense FP32 matrix multiplication in **row-major** order.
For performance, use **shared-memory tiling** when performing matrix multiplications on the GPU.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be written to the **output** array in row-major order

## Input Description

You will be given two integers `N` and `P`, followed by `N×N` floating-point numbers for matrix `A` in row-major order.

Input format:

```bash
N P
A_11 A_12 ... A_1N
A_21 A_22 ... A_2N
...
A_N1 A_N2 ... A_NN
```

Constraints:

* 1 ≤ N ≤ 1024
* 1 ≤ P ≤ 20
* −10.0 ≤ Aᵢⱼ ≤ 10.0

Solutions are tested with:

* Absolute tolerance ≤ 1e−2
* Relative tolerance ≤ 1e−2

## Output Description

Output the resulting matrix (A^P) in **row-major order**, formatted as `N` lines with `N` floating-point numbers per line.

Output format:

```bash
O_11 O_12 ... O_1N
O_21 O_22 ... O_2N
...
O_N1 O_N2 ... O_NN
```

## Example

### Input

```
3 2
1 2 3
4 5 6
7 8 9
```

### Output

```
30.0000 36.0000 42.0000
66.0000 81.0000 96.0000
102.0000 126.0000 150.0000
```

## How to Run

### 1. Build the Program

```bash
cd hard/matrix_power
make
```

Or build from the top-level hard directory:

```bash
cd hard
make matrix_power
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
echo "3 2
1 2 3
4 5 6
7 8 9" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
