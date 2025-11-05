# Histogram

## Description

Implement a program that computes the **histogram** of an array of integers on a GPU using parallel computing techniques.
The program should take an input array and produce an output histogram where each bin represents the count of elements with that specific value.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* Must use GPU parallel computing for efficiency
* The final result must be stored in the histogram array

## Input Description

You will be given two integers `N` and `num_bins`, followed by `N` integer values.

Input format:

```bash
N num_bins
a1 a2 ... aN
```

Constraints:

* 1 ≤ N ≤ 100,000,000 (integer)
* 1 ≤ num_bins ≤ 1024 (integer)
* 0 ≤ aᵢ < num_bins (integer)

## Output Description

Output `num_bins` integer numbers representing the histogram result, separated by spaces, with a newline at the end.

Output format:

```bash
count(0) count(1) ... count(num_bins-1)
```

Where
`count(i)` = number of elements in the input array that have value `i`.

## Example

### Input

```
8 4
0 1 2 3 0 1 2 0
```

### Output

```
3 2 2 1
```

## How to Run

### 1. Build the Program

```bash
cd medium/histogram
make
```

Or build from the top-level medium directory:

```bash
cd medium
make histogram
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
echo "8 4
0 1 2 3 0 1 2 0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
