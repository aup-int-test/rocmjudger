# RainBow Table

## Description

Implement a program that applies the **FNV1a hash function** to an array of 32-bit integers on a GPU.
The program should take an input array and a number of iterations `R`, then produce an output array where each element is hashed `R` times using the FNV1a algorithm.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array

## Input Description

You will be given two integers `N` and `R`, followed by `N` integer values.

Input format:

```bash
N R
a1 a2 ... aN
```

Constraints:

* 1 ≤ N ≤ 10,000,000, Length of array (integer)
* 1 ≤ R ≤ 1,000, Number of iterations (integer)
* −2³¹ ≤ aᵢ ≤ 2³¹−1, Array values (integer)

## Output Description

Output `N` unsigned integer values representing the FNV1a hash results after `R` iterations, separated by spaces, with a newline at the end.

Output format:

```bash
hash(a1) hash(a2) ... hash(aN)
```

Where each `hash(aᵢ)` is the result of applying the FNV1a hash function `R` times to `aᵢ`.

## Example

### Input

```
3 2
123 456 789
```

### Output

```
3728671011 2847294259 1965917507
```

## How to Run

### 1. Build the Program

```bash
cd medium/rainbow_table
make
```

Or build from the top-level medium directory:

```bash
cd medium
make rainbow_table
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
echo "3 2
123 456 789" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
