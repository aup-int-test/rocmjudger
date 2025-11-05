# Matrix Transpose

## Description

Implement a program that performs **matrix transpose** of a matrix containing 32-bit floating point numbers on a GPU.
The program should take an input matrix `A` of size `(rows × cols)` and produce an output matrix where
( A^T[j][i] = A[i][j] ).

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array

## Input Description

You will be given 2 values: `rows` and `cols`, followed by `rows × cols` floating point values for the matrix.

Input format:

```bash
rows cols
a11 a12 ... a1_cols
a21 a22 ... a2_cols
...
a_rows1 a_rows2 ... a_rows_cols
```

Constraints:

* 1 ≤ rows, cols ≤ 3000 (integer)
* aᵢⱼ: Matrix values (float)

## Output Description

Output `rows × cols` floating point numbers representing the **transposed matrix**, separated by spaces, with each row on a new line.

Output format:

```bash
a11 a21 ... a_rows1
a12 a22 ... a_rows2
...
a1_cols a2_cols ... a_rows_cols
```

Where the output matrix has dimensions `(cols × rows)` and `output[j][i] = input[i][j]`.

## Example

### Input

```
2 3
1.0 2.0 3.0
4.0 5.0 6.0
```

### Output

```
1 4
2 5
3 6
```

## How to Run

### 1. Build the Program

```bash
cd medium/matrix_transpose
make
```

Or build from the top-level medium directory:

```bash
cd medium
make matrix_transpose
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
echo "2 3
1.0 2.0 3.0
4.0 5.0 6.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
