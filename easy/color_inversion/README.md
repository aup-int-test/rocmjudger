# Color Inversion

## Description

Implement a program that performs **image color inversion** on a GPU.
The program should take an input image represented as RGBA pixels and produce an output image where each RGB channel is inverted (`255 - original_value`).
The alpha channel remains unchanged.

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the image array

## Input Description

You will be given 2 integers `width` and `height`, followed by `width × height × 4` integer values representing RGBA pixel data.

Input format:

```bash
width height
r1 g1 b1 a1 r2 g2 b2 a2 ... r_n g_n b_n a_n
```

Constraints:

* 1 ≤ width, height ≤ 10,000 (integer)
* 0 ≤ rᵢ, gᵢ, bᵢ, aᵢ ≤ 255 (unsigned char)

## Output Description

Output `width × height × 4` integer values representing the inverted RGBA pixel data,
with **4 values per line** (one pixel per line).

Output format:

```bash
inverted_r1 inverted_g1 inverted_b1 a1
inverted_r2 inverted_g2 inverted_b2 a2
...
inverted_r_n inverted_g_n inverted_b_n a_n
```

Where:

* `inverted_r = 255 - r`
* `inverted_g = 255 - g`
* `inverted_b = 255 - b`
* `a` (alpha) remains unchanged.

## Example

### Input

```
2 1
100 150 200 255 50 75 25 128
```

### Output

```
155 105 55 255
205 180 230 128
```

## How to Run

### 1. Build the Program

```bash
cd medium/color_inversion
make
```

Or build from the top-level medium directory:

```bash
cd medium
make color_inversion
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
echo "2 1
100 150 200 255 50 75 25 128" | ./main
```

Or redirect from file:

```bash
./main < testcases/1.in
```

### 4. Clean Up

```bash
make clean
```
