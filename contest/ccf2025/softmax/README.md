# Softmax

## Description

Implement a GPU program that computes the **softmax** of a 1-D array of floating-point numbers.
Given an input vector $\mathbf{x} = [x_1, x_2, \dots, x_N]$, produce $\mathbf{y} = [y_1, y_2, \dots, y_N]$ where:

$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

Because naive exponentiation can overflow/underflow, you **must** use the numerically stable form:

$$
m = \max_i x_i, \quad
t_i = e^{x_i - m}, \quad
S = \sum_{i=1}^{N} t_i, \quad
y_i = \frac{t_i}{S}
$$

## Requirements

* The `solve` function signature must remain unchanged.
* Use the numerically stable formulation above.
* Only a **single-GPU** implementation is allowed (no multi-GPU).

## Code Structure

```
.
├── main.cpp        # Reads input, calls solve(), prints result
├── kernel.hip      # GPU kernels + solve() implementation
├── main.h          # Shared includes + solve() declaration
├── Makefile
├── README.md
└── testcases.zip   # Sample testcases for local verification
```

## Build & Run

### Build

```bash
make
```

Produces executable: `softmax`.

### Run

```bash
./softmax input.txt
```

---

## Testcases

The `testcases/` folder contains **15** sample input files and corresponding outputs.

Run a sample as:

```bash
./softmax testcases/1.in
```

Tolerances:

* Absolute tolerance: $1\times 10^{-6}$
* Relative tolerance: $1\times 10^{-5}$
* Minimum denominator: $1\times 10^{-12}$

---

### Input Format

* The first line contains a single integer $N$, the length of the array.
* The second line contains $N$ floating-point numbers separated by spaces.

**Example**

```
3
1.0 2.0 3.0
```

**Constraints**

* $1 \le N \le 100{,}000{,}000$
* $\text{input}[i]$ are floating-point numbers

---

### Output Format

* Output $N$ floating-point numbers representing the **softmax** values $y_1, y_2, \dots, y_N$.
* Each number should satisfy the given tolerance requirements.
* Numbers are separated by spaces and followed by a newline.

**Example**

```
0.090 0.244 0.665
```

---

## Submission

Your submitted folder must named `softmax`

Contain all required source files (`main.cpp`, `kernel.hip`, `main.h`, `Makefile`) so that it can be built directly with:

```bash
make
```

The grader should be able to:

```bash
cd $HOME/hip_programming_contest/softmax
make
./softmax <hidden_testcase.txt>
```

---