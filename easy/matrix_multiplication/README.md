# Matrix Multiplication

## Description

Implement a program that performs **matrix multiplication** of two matrices containing 32-bit floating point numbers on a GPU.
The program should take two input matrices `A (M×N)` and `B (N×K)` and produce an output matrix `C (M×K)` containing their product.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in matrix `C`

## Input Description

You will be given 3 values `M`, `N`, and `K`, followed by `M×N` values for matrix `A` and `N×K` values for matrix `B`.

Input format:

```bash
M N K
a11 a12 ... a1N
a21 a22 ... a2N
...
aM1 aM2 ... aMN
b11 b12 ... b1K
b21 b22 ... b2K
...
bN1 bN2 ... bNK
```

Constraints:

* 1 ≤ M, N, K ≤ 1000 (integer)
* aᵢⱼ, bᵢⱼ: Matrix values (float)

## Output Description

Output `M×K` floating point numbers representing the matrix multiplication result, formatted to 3 decimal places and separated by spaces, with each row on a new line.

Output format:

```bash
c11 c12 ... c1K
c21 c22 ... c2K
...
cM1 cM2 ... cMK
```

Where
[
c_{ij} = \sum_{k=1}^{N} a_{ik} \times b_{kj}
]
for i = 1, 2, ..., M and j = 1, 2, ..., K.

## Example

### Input

```
2 3 2
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0
9.0 10.0
11.0 12.0
```

### Output

```
58.000 64.000
139.000 154.000
```

## How to Run

### 1. Build the Program

```bash
cd medium/matrix_multiplication
make
```

Or build from the top-level medium directory:

```bash
cd medium
make matrix_multiplication
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
echo "2 3 2
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0
9.0 10.0
11.0 12.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
