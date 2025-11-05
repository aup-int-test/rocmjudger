# Monte Carlo Integration

## Description

Implement a program that performs **Monte Carlo numerical integration** on a GPU using **shared-memory parallel reduction**.
The program takes integration bounds `a`, `b`, the number of samples `n_samples`, and an array of `n_samples` double values (the sampled y-values), then outputs the approximate integral.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the `result` variable

## Input Description

You will be given 3 values: `a`, `b`, and `n_samples`, followed by `n_samples` double values.

Input format:

```bash
a b n_samples
y1 y2 ... y_n_samples
```

Constraints:

* −1,000,000 ≤ a, b ≤ 1,000,000 (double)
* 1 ≤ n_samples ≤ 100,000,000 (integer)
* yᵢ: Sample y-values (double)

## Output Description

Output a single double value representing the Monte Carlo integration result.

Output format:

```bash
result
```

Where `result ≈ (b − a) × (1/n) × Σ yᵢ`.

## Example

### Input

```
0 2 8
0.0625 0.25 0.5625 1.0 1.5625 2.25 3.0625 4.0
```

### Output

```
3.125000
```

## How to Run

### 1. Build the Program

```bash
cd medium/monte_carlo_integration
make
```

Or build from the top-level medium directory:

```bash
cd medium
make monte_carlo_integration
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
echo "0 2 8
0.0625 0.25 0.5625 1.0 1.5625 2.25 3.0625 4.0" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
