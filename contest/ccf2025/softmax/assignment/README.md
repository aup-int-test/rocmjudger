# Softmax

## Description

# Problem: Softmax on GPU — From Research Notebook to Real-Time Inference

Your team just shipped a prototype vision model to a demo kiosk. It works—until crowds show up. As inputs stream in, the CPU softmax becomes the bottleneck: latency spikes, frames drop, and the “smart” kiosk stops feeling smart. Your task is to move this last mile onto the GPU and make it rock-solid and fast.

## Goal

Implement a GPU program that computes the **softmax** of a 1-D array of floating-point numbers. Given an input vector $\mathbf{x} = [x_1, x_2, \dots, x_N]$, produce $\mathbf{y} = [y_1, y_2, \dots, y_N]$ where

$$
y_i \;=\; \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}.
$$

Because naive exponentiation can overflow/underflow, you **must** use the numerically stable form:

$$
m \;=\; \max_i x_i,\qquad
t_i \;=\; e^{\,x_i - m},\qquad
S \;=\; \sum_{i=1}^{N} t_i,\qquad
y_i \;=\; \frac{t_i}{S}.
$$

## Requirements

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array
* Use the numerically stable formulation 

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

## Submission

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



