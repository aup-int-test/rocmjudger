# Prefix Sum

## DESCRIPTION

Your task is to implement a GPU-accelerated program that computes the prefix sum of an array of integers efficiently. The program should take an input array of integers—potentially containing millions or even hundreds of millions of elements—and produce an output array where each element is the sum of all preceding values up to that position.

## REQUIREMENTS

* External libraries are not permitted
* The `solve` function signature must remain unchanged
* The final result must be stored in the output array


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

Produces executable: `prefix_sum`

### Run

```bash
./prefix_sum input.txt
```

---

## TESTCASES
The testcases/ folder contains 15 sample input files.

You may run them as:

```bash
./prefix_sum testcases/1.in
```

Hidden testcases will be used during grading, so ensure your solution handles.

Each file follows the input format described below.

### Input

```
N
a1 a2 ... aN
```

Constraints:

* `1 ≤ N ≤ 1000000000`
* `-1000 ≤ ai ≤ 1000`

### Output
`prefix_sum(a1) prefix_sum(a2) ... prefix_sum(aN)\n`

Where `prefix_sum(ai) = a1 + a2 + ... + ai`.

---

## SUBMISSION

Your submitted folder must:

Contain all required source files (`main.cpp`, `kernel.hip`, `main.h`, `Makefile`) so that it can be built directly with:

```bash
make
```

The grader should be able to:

```bash
cd <submission-folder>
make
./prefix_sum <hidden_testcase.txt>
```

## Hint: Blocked Prefix Sum Algorithm

Given an input array $A[0 \dots n-1]$, the prefix sum problem computes an output array $S[0 \dots n-1]$ where:

$$
S[i] = \sum_{j=0}^{i} A[j] \quad \text{(inclusive scan)}
$$

or, for the exclusive form:

$$
S[i] = \sum_{j=0}^{i-1} A[j]
$$

A blocked (or tiled) prefix sum algorithm partitions the input array into $M$ blocks, each containing $B$ consecutive elements (the **blocking factor**). The algorithm proceeds in three main phases:

**Phase 1: Local scan within each block**
Each block $k$ independently computes the prefix sum of its elements (using shared memory), producing a *local scan*. The total sum of block $k$ (the last element of its local scan) is written to `blockSums[k]`. This step is fully parallel across blocks.

**Phase 2: Scan of block sums**
The `blockSums` array is scanned to produce `blockOffsets`, where `blockOffsets[k]` is the total sum of all elements in preceding blocks $0 \dots k-1$. This is typically an **exclusive** scan.

**Phase 3: Offset addition to local results**
For each block $k$, all elements in its local scan are incremented by `blockOffsets[k]` to form the final global prefix sums.

**Example**
Let $A = [2, 1, 3, 4, 2, 1, 5, 0]$, $B = 4$.

* Phase 1 (local scan):
  Block 0 → `[2, 3, 6, 10]`, blockSum = 10
  Block 1 → `[2, 3, 8, 8]`, blockSum = 8
* Phase 2 (scan of block sums): `blockOffsets = [0, 10]`
* Phase 3 (offset addition): Block 1 → `[12, 13, 18, 18]`

Final result $S = [2, 3, 6, 10, 12, 13, 18, 18]$.

---

