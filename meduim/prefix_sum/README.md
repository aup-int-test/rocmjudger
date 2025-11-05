# Prefix Sum (Parallel Scan)

## Description

Implement a program that computes the **prefix sum (parallel scan)** of an array of integers on a GPU using **hierarchical scanning**.
The program should take an input array and produce an output array where each element is the sum of all previous elements including itself.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* Must use hierarchical parallel scanning for efficiency
* The final result must be stored in the output array

## Input Description

You will be given one integer `N`, followed by `N` integer values.

Input format:

```bash
N
a1 a2 ... aN
```

Constraints:

* 1 ≤ N ≤ 100,000,000 (integer)
* −1000 ≤ aᵢ ≤ 1000 (integer)

## Output Description

Output `N` integers representing the prefix sum result, separated by spaces, with a newline at the end.

Output format:

```bash
prefix_sum(a1) prefix_sum(a2) ... prefix_sum(aN)
```

Where
[
\text{prefix_sum}(a_i) = a_1 + a_2 + \dots + a_i
]

for each position `i`.

## Example

### Input

```
5
1 2 3 4 5
```

### Output

```
1 3 6 10 15
```

## How to Run

### 1. Build the Program

```bash
cd medium/prefix_sum
make
```

Or build from the top-level medium directory:

```bash
cd medium
make prefix_sum
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
