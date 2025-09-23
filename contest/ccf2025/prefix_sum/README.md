# Prefix Sum

## Description

Your task is to implement a GPU-accelerated program that computes the **prefix sum** of an array of integers efficiently.
The program should take an input array of integers—potentially containing millions or even hundreds of millions of elements—and produce an output array where each element is the sum of all preceding values up to that position.

## Requirements

* The `solve` function signature must remain unchanged.
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

Produces executable: `prefix_sum`.

### Run

```bash
./prefix_sum input.txt
```

---

## Testcases

The `testcases/` folder contains **15** sample input files and output files.

You may run them as:

```bash
./prefix_sum testcases/1.in
```

Hidden testcases will be used during grading, so ensure your solution handles large inputs and edge cases.

---

### Input Format

* The first line contains a single integer $N$, the length of the array.
* The second line contains $N$ space-separated integers representing the array values.

**Example**

```
5
1 2 3 4 5
```

**Constraints**

* $1 \le N \le 1{,}000{,}000{,}000$
* $-1000 \le \text{input}[i] \le 1000$

---

### Output Format

* Output $N$ space-separated integers representing the prefix sums in order.
* Use an **inclusive scan**:

$$
S[i] = \sum_{j=0}^{i} A[j]
$$

* End the output with a newline character.

**Example**

```
1 3 6 10 15
```

---

## Submission

Your submitted folder must named `prefix_sum`:

Contain all required source files (`main.cpp`, `kernel.hip`, `main.h`, `Makefile`) so that it can be built directly with:

```bash
make
```

The grader should be able to:

```bash
cd $HOME/hip_programming_contest/prefix_sum
make
./prefix_sum <hidden_testcase.txt>
```

---

## Hint: Blocked Prefix Sum Algorithm

Given an input array $A[0 \dots n-1]$, the prefix sum problem computes an output array $S[0 \dots n-1]$ where:

$$
S[i] = \sum_{j=0}^{i} A[j] \quad \text{(inclusive scan)}
$$

or, for the exclusive form:

$$
S[i] = \sum_{j=0}^{i-1} A[j]
$$

A **blocked** (or tiled) prefix sum algorithm partitions the input array into $M$ blocks, each containing $B$ consecutive elements (the **blocking factor**). The algorithm proceeds in three main phases:

1. **Local scan within each block**
   Each block $k$ independently computes the prefix sum of its elements (using shared memory), producing a *local scan*.
   The total sum of block $k$ (the last element of its local scan) is written to $\text{blockSums}[k]$.
   This step is fully parallel across blocks.

2. **Scan of block sums**
   The $\text{blockSums}$ array is scanned to produce $\text{blockOffsets}$, where $\text{blockOffsets}[k]$ is the total sum of all elements in preceding blocks $0 \dots k-1$.
   This is typically an **exclusive** scan.

3. **Offset addition to local results**
   For each block $k$, all elements in its local scan are incremented by $\text{blockOffsets}[k]$ to form the final global prefix sums.

---
