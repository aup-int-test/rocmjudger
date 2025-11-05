# Array Reduction

## Description

Implement a program that performs **parallel reduction (sum)** of an array of 32-bit floating point numbers on a GPU.
The program should take an input array and compute the sum of all elements using GPU parallel reduction techniques.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output variable

## Input Description

You will be given one integer `N`, followed by `N` floating point values.

Input format:

```bash
N
a1 a2 ... aN
```

Constraints:

* 1 ≤ N ≤ 1,000,000,000 (integer)
* aᵢ: Array values (float)

## Output Description

Output a single floating point number representing the sum of all array elements.

Output format:

```bash
sum
```

Where
[
\text{sum} = a_1 + a_2 + \dots + a_N
]

## Example

### Input

```
5
1 2 3 4 5
```

### Output

```
15
```

## How to Run

### 1. Build the Program

```bash
cd medium/array_reduction
make
```

Or build from the top-level medium directory:

```bash
cd medium
make array_reduction
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
echo "5
1 2 3 4 5" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
