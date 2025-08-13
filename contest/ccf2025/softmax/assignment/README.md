# Softmax

## Description

Implement a program that computes the softmax of an array of floating-point numbers on a GPU. The program should take an input array and produce an output array.

## Requirements

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array
* Use the numerically stable formulation with **max subtraction**

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


## Testcases

The `testcases/` folder contains 15 sample input files and output files.

You may run them as:

```bash
./softmax testcases/1.in
```

Tolerances: 
absolute tolerance: 1e-8
relative tolerance: 1e-6
minimum denominator: 1e-12

### Input

```
3
1.0, 2.0, 3.0
```

Constraints:

* `1 ≤ N ≤ 100000000`
* `input[i]` are floating-point numbers

### Output
```
0.090, 0.244, 0.665
```

### Submission

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

---

## Hint: Softmax Algorithm

Given an input vector

$$
x = [x_1, x_2, \dots, x_N],
$$

compute:

1. Maximum value:

$$
m = \max_{1 \leq i \leq N} x_i
$$

2. Exponentiation with max subtraction:

$$
t_i = e^{x_i - m}, \quad \forall i \in \{1, 2, \dots, N\}
$$

3. Sum of all exponentials:

$$
S = \sum_{i=1}^N t_i
$$

4. Softmax output:

$$
y_i = \frac{t_i}{S}, \quad \forall i \in \{1, 2, \dots, N\}
$$

This **max-subtraction** form avoids overflow/underflow and improves numerical stability.

