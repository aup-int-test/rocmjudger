# 2D Max Pooling

## Description

Implement a **2D Max Pooling** operation for image or feature map downsampling in **FP32**.
Given an input tensor **input** of shape `[N, C, H, W]`, apply max pooling with the specified **kernel size**, **stride**, and **padding**, and produce an output tensor **output** of shape `[N, C, H_out, W_out]`.

For each output position `(n, c, h_out, w_out)`, compute:

[
output[n, c, h_{out}, w_{out}] =
\max_{0 \le i < kernel_size, 0 \le j < kernel_size}
input[n, c, h_{out} \times stride + i - padding, w_{out} \times stride + j - padding]
]

All tensors use **row-major order** and contain **32-bit floating point** values.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the **output** tensor

## Input Description

The input contains seven integers and the input tensor data:

```bash
N C H W kernel_size stride padding
input_1 input_2 ... input_{N×C×H×W}
```

Constraints:

* 1 ≤ N, C ≤ 8
* 2 ≤ H, W ≤ 1024
* 1 ≤ kernel_size ≤ min(H, W)
* 1 ≤ stride ≤ kernel_size
* 0 ≤ padding ≤ kernel_size / 2
* All values are FP32

## Output Description

Output `N × C × H_out × W_out` floating point numbers representing the result tensor, where:

[
H_{out} = \frac{H + 2 \times padding - kernel_size}{stride} + 1
]

[
W_{out} = \frac{W + 2 \times padding - kernel_size}{stride} + 1
]

Each channel and batch should be printed in row-major order, with spaces between values of a row and newlines separating rows.
Leave an empty line between channels and batches.

Output format:

```bash
output[n=0, c=0]
y_0_0 y_0_1 ... y_0_Wout
...
(repeat for all n, c)
```

## Example

### Input

```
1 1 3 3 2 1 0
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
```

### Output

```
5.000000 6.000000
8.000000 9.000000
```

## How to Run

### 1. Build the Program

```bash
cd medium/max_pooling_2d
make
```

Or build from the top-level medium directory:

```bash
cd medium
make max_pooling_2d
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
echo "1 1 3 3 2 1 0
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
