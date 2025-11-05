# 2D Convolution

## Description

Implement a program that performs **2D convolution** of an input matrix with a kernel matrix on a GPU.
The program should take an input matrix and a kernel matrix, then produce an output matrix containing the convolution result.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the **output array**

## Input Description

You will be given four integers:
`input_rows`, `input_cols`, `kernel_rows`, `kernel_cols`,
followed by the **input matrix** and **kernel matrix** (both in row-major order).

Input format:

```bash
input_rows input_cols kernel_rows kernel_cols
a11 a12 ... a1_input_cols
a21 a22 ... a2_input_cols
...
a_input_rows1 a_input_rows2 ... a_input_rows_input_cols
k11 k12 ... k1_kernel_cols
k21 k22 ... k2_kernel_cols
...
k_kernel_rows1 k_kernel_rows2 ... k_kernel_rows_kernel_cols
```

Constraints:

* 1 ≤ input_rows, input_cols ≤ 3072
* 1 ≤ kernel_rows, kernel_cols ≤ 64
* kernel_rows ≤ input_rows, kernel_cols ≤ input_cols
* aᵢⱼ, kᵢⱼ are 32-bit floating-point numbers

## Output Description

Output `(input_rows - kernel_rows + 1) × (input_cols - kernel_cols + 1)` floating point numbers representing the convolution result, with each row on a new line and values separated by spaces.

Output format:

```bash
c11 c12 ... c1_output_cols
c21 c22 ... c2_output_cols
...
c_output_rows1 c_output_rows2 ... c_output_rows_output_cols
```

Where
[
c_{ij} = \sum_{m=0}^{kernel_rows-1} \sum_{n=0}^{kernel_cols-1} a_{(i+m)(j+n)} \cdot k_{mn}
]

## Example

### Input

```
2 3 2 3
1. 2. 3.
4. 5. 6.
1. 0.
```

### Output

```
32.
```

## How to Run

### 1. Build the Program

```bash
cd medium/convolution_2d
make
```

Or build from the top-level medium directory:

```bash
cd medium
make convolution_2d
```

To compile for a specific GPU architecture:

```bash
make GPU_ARCH=gfx90a    # For AMD MI210
make GPU_ARCH=gfx908    # For AMD MI100
make GPU_ARCH=gfx1100   # For AMD Radeon W7900
```

This builds the `main` executable from the combined `main.hip` source.

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
echo "2 3 2 3
1. 2. 3.
4. 5. 6.
1. 0." | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
