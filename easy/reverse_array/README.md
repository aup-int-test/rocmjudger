# Array Reverse

## Description

Implement a program that reverses an array of 32-bit floating point numbers on a GPU. The program should take an input array and reverse the order of its elements in-place.

* External libraries are not permitted
* The solve function signature must remain unchanged
* The final result must be stored in the input array

## Input Description

You will be given 1 value N, followed by N floating point values.

Input format:

```bash
N
a1 a2 ... aN
```

Constraints:

* 1 ≤ N ≤ 10000000, Length of array(integer)
* ai, Array values(float)

## Output Description

Output N floating point numbers representing the reversed array, separated by spaces, with a newline at the end.

Output format:

```bash
aN aN-1 ... a1
```

Where the output array contains the same elements as the input array but in reverse order.

## Example

### Input

```
5
1.0 2.0 3.0 4.0 5.0
```

### Output

```
5.0 4.0 3.0 2.0 1.0
```

## How to Run

### 1. Build the Program

```bash
cd easy/array_reverse
make
```

Or build from the top-level easy directory:

```bash
cd easy
make array_reverse
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
1.0 2.0 3.0 4.0 5.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
