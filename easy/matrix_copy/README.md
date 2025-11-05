# Matrix Copy

## Description

Implement a program that copies a **square matrix** containing 32-bit floating point numbers on a GPU.
The program should take an input matrix `A (N×N)` and copy it to matrix `B` using GPU operations.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The operation must be performed on the GPU

## Input Description

You will be given one integer `N`, followed by `N×N` floating point values for the matrix.

Input format:

```bash
N
a11 a12 ... a1N
a21 a22 ... a2N
...
aN1 aN2 ... aNN
```

Constraints:

* 1 ≤ N ≤ 10,000 (integer)
* aᵢⱼ: Matrix values (float)

## Output Description

Output `N×N` floating point numbers representing the copied matrix, with each row on a new line and values separated by spaces.

Output format:

```bash
a11 a12 ... a1N
a21 a22 ... a2N
...
aN1 aN2 ... aNN
```

Where the output is identical to the input matrix.

## Example

### Input

```
2
1.5 2.3
4.1 5.9
```

### Output

```
1.5 2.3
4.1 5.9
```

## How to Run

### 1. Build the Program

```bash
cd easy/matrix_copy
make
```

Or build from the top-level easy directory:

```bash
cd easy
make matrix_copy
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
echo "2
1.5 2.3
4.1 5.9" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
