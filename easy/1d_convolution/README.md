# 1D Convolution

## Description

Implement a program that performs **1D convolution** of an input signal with a kernel on a GPU.
The program should take an input vector and a kernel vector, then produce an output vector containing the convolution result.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array

## Input Description

You will be given 2 integers `input_size` and `kernel_size`, followed by `input_size` floating point input values and `kernel_size` floating point kernel values.

Input format:

```bash
input_size kernel_size
a1 a2 ... a_input_size
k1 k2 ... k_kernel_size
```

Constraints:

* 1 ≤ input_size ≤ 1,000,000
* 1 ≤ kernel_size ≤ input_size
* aᵢ: Input array values (float)
* kⱼ: Kernel array values (float)

## Output Description

Output `(input_size - kernel_size + 1)` floating point numbers representing the 1D convolution result, separated by spaces, with a newline at the end.

Output format:

```bash
c1 c2 ... c_output_size
```

Where
[
c_i = \sum_{j=0}^{\text{kernel_size}-1} a_{i+j} \times k_j
]

for i = 0, 1, ..., output_size−1.

## Example

### Input

```
5 3
1.0 2.0 3.0 4.0 5.0
0.5 1.0 0.5
```

### Output

```
4 6 8
```

## How to Run

### 1. Build the Program

```bash
cd medium/1d_convolution
make
```

Or build from the top-level medium directory:

```bash
cd medium
make 1d_convolution
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
echo "5 3
1.0 2.0 3.0 4.0 5.0
0.5 1.0 0.5" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
