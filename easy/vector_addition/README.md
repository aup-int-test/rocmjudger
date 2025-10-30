# Vector Addition

## Description
Implement a program that performs element-wise addition of two vectors containing 32-bit floating point numbers on a GPU. The program should take two input vectors of equal length and produce a single output vector containing their sum.

- External libraries are not permitted
- The solve function signature must remain unchanged
- The final result must be stored in vector C

## Input Description
You will be given 1 value N, followed by N function values.

Input format:
```bash
N  
a1, a2, ... an  
b1, b2, ... bn
```

Constraints:
- 1 ≤ N ≤ 1000000, Length of array(integer)
- a1, a2, ... an, Array values(float)
- b1, b2, ... bn, Array values(float)

## Output Description
Output N floating point numbers representing the element-wise sum of the two input vectors, formatted to 3 decimal places and separated by spaces, with a newline at the end.

Output format:
```bash
c1 c2 ... cn
```

Where ci = ai + bi for i = 1, 2, ..., n

## Example
### Input
```
5
1.5 2.3 3.7 4.1 5.9
0.8 1.2 2.3 3.4 4.6
```

### Output
```
2.300 3.500 6.000 7.500 10.500
```

## How to Run

### 1. Build the Program
```bash
cd easy/vector_addition
make
```

Or build from the top-level easy directory:
```bash
cd easy
make vector_addition
```

To compile for a specific GPU architecture:
```bash
make GPU_ARCH=gfx90a    # For AMD MI210
make GPU_ARCH=gfx908    # For AMD MI100
make GPU_ARCH=gfx1100   # For AMD Radeon W7900
```

### 2. Generate Test Cases (Optional)
```bash
python3 geninput.py
```

### 3. Run the Program

**Option A: Interactive input (stdin)**
```bash
./exe_main
```
Then enter the input manually.

**Option B: File input**
```bash
./exe_fs_main testcases/1.in
```

**Option C: Pipe input**
```bash
echo "5
1.5 2.3 3.7 4.1 5.9
0.8 1.2 2.3 3.4 4.6" | ./exe_main
```

Or redirect from file:
```bash
./exe_main < testcases/1.in
```

### 4. Clean Up
```bash
make clean
```
