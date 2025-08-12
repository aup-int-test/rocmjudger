# Softmax

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

Produces executable: `softmax`

### Run

```bash
./softmax input.txt
```

## I/O Format

**Input**

```
N
a1 a2 ... aN
```

Constraints:

* `1 ≤ N ≤ 100000000`
* `a_i` are floating-point numbers

**Output**
`softmax(a1) softmax(a2) ... softmax(aN)\n`

---

### DESCRIPTION

Implement a program that computes the softmax of an array of floating-point numbers on a GPU. The program should take an input array and produce an output array.

### REQUIREMENTS

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array
* Use the numerically stable formulation with **max subtraction**

### TESTCASES

The `testcases/` folder contains six sample input files.

Each file follows the input format described above.

You may run them as:

```bash
./softmax testcases/1.in
```

Tolerances: 
absolute tolerance: 1e-8
relative tolerance: 1e-6
minimum denominator: 1e-12

### SUBMISSION

Your submitted folder must:

Contain all required source files (`main.cpp`, `kernel.hip`, `main.h`, `Makefile`) so that it can be built directly with:

```bash
make
```

The grader should be able to:

```bash
cd <submission-folder>
make
./softmax <hidden_testcase.txt>
```

## Hint: Softmax Algorithm

Given an input vector x = \[x1, ..., xN], compute:

* m = max(x)
* t\_i = exp(x\_i − m) for each i
* S = sum of all t\_i
* y\_i = t\_i / S for each i

This max-subtraction form avoids overflow/underflow and improves numerical stability.
