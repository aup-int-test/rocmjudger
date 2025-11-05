# Leaky ReLU

## Description

Implement a program that applies the **Leaky ReLU activation function** to an array of 32-bit floating point numbers on a GPU.
The program should take an input array and produce an output array where each element is transformed by the Leaky ReLU function.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array

## Input Description

You will be given one integer `N`, followed by `N` floating point values.

Input format:

```bash
N
a1 a2 ... aN
```

Constraints:

* 1 ≤ N ≤ 10,000,000 (integer)
* aᵢ: Array values (float)

## Output Description

Output `N` floating point numbers representing the Leaky ReLU activation result, separated by spaces, with a newline at the end.

Output format:

```bash
leaky_relu(a1) leaky_relu(a2) ... leaky_relu(aN)
```

Where
[
\text{LeakyReLU}(x) =
\begin{cases}
x, & x > 0 \
0.01x, & x \le 0
\end{cases}
]

## Example

### Input

```
6
-2.5 1.3 0.0 -5.7 3.8 -1.0
```

### Output

```
-0.025 1.3 0 -0.057 3.8 -0.01
```

## How to Run

### 1. Build the Program

```bash
cd easy/leaky_relu
make
```

Or build from the top-level easy directory:

```bash
cd easy
make leaky_relu
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
echo "6
-2.5 1.3 0.0 -5.7 3.8 -1.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
