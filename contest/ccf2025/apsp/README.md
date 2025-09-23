# All-Pairs Shortest Path (APSP)

## Description

Implement an efficient **all-pairs shortest path (APSP)** solver on GPU.

Given a directed, weighted graph with **non-negative** edge weights, compute the shortest-path distance from every vertex $i$ to every vertex $j$.

## Requirements

* You may choose **any** algorithm to solve APSP.
* You must implement the shortest-path algorithm **yourself** (no external libraries).
* Only a **single-GPU** implementation is allowed (no multi-GPU).

## Code Structure

```
.
├── main.cpp
├── main.h
├── Makefile
├── README.md
└── testcases.zip   # Sample testcases for local verification
```

## Build & Run

### Build

```bash
make
```

Produces executable: `apsp`.

### Run

```bash
./apsp input.txt
```

---

## Testcases

The `testcases/` folder contains **15** sample input files and corresponding outputs.

Run a sample as:

```bash
./apsp testcases/1.in
```

Hidden testcases will be used during grading; ensure your solution handles edge cases and large graphs.

---

### Input Format

* The graph is directed with non-negative edge weights.
* All values are **32-bit integers** (use `int` in C/C++).
* The first two integers are the number of vertices and edges: $(V, E)$.
* Then follow $E$ edges; each edge is given by three integers:

$$
\mathrm{src}_i\ \ \mathrm{dst}_i\ \ \mathrm{w}_i \quad\text{for } i=0,1,\dots,E-1 .
$$

* Vertex IDs are $0,1,\dots,V-1$.

**Example**

```
2 1
0 1 5
```

**Constraints**

* $2 \le V \le 40{,}000$
* $0 \le E \le V \times (V-1)$
* $0 \le \mathrm{src}_i, \mathrm{dst}_i < V$
* $\mathrm{src}_i \ne \mathrm{dst}_i$ (no self-loops in the input)
* If $\mathrm{src}_i=\mathrm{src}_j$ then $\mathrm{dst}_i \ne \mathrm{dst}_j$ (no duplicate edges with the same $\mathrm{src}$ and $\mathrm{dst}$)
* $0 \le \mathrm{w}_i \le 1000$

---

### Output Format

You must print $V^2$ integers to **standard output** representing the distance matrix $D$ where:

$$
D[i,j] = d(i,j)
$$

is the shortest-path distance from vertex $i$ to vertex $j$.

* Distances must be printed in **row-major order by source vertex**:

$d(0,0),\, d(0,1),\, \ldots,\, d(0,V-1);\quad
d(1,0),\, \ldots,\, d(1,V-1);\quad \ldots;\quad
d(V-1,0),\, \ldots,\, d(V-1,V-1).$

* Diagonal entries must satisfy:

$$
d(i,i) = 0 \quad \forall\, i .
$$

* If there is **no path** from $i \to j$, output:

$$
d(i,j) = 2^{30} - 1 = 1073741823 .
$$

**Example**

```
0 5
1073741823 0
```

---

## Submission

Your submitted folder must named `apsp`:

Contain all required source files (`main.cpp`, `main.h`, `Makefile`) so that it can be built directly with:

```bash
make
```

The grader will test with:

```bash
cd $HOME/hip_programming_contest/apsp
make
./apsp <hidden_testcase.txt>
```

Ensure your program reads the input as specified and prints exactly $V^2$ integers in the required order.

---

## Hint: Blocked Floyd–Warshall Algorithm
Let $B$ be the block size. The $V \times V$ distance matrix is divided into $\lceil V/B \rceil \times \lceil V/B \rceil$ square tiles of size $B \times B$.

For each block index $k$ (from $0$ to $\lceil V/B \rceil - 1$):

1. **Update the pivot block** $(k,k)$ — compute shortest paths within the pivot block considering vertices inside it as intermediates.
2. **Update pivot row and pivot column blocks** — update distances in the same row or column as the pivot using newly computed pivot distances.
3. **Update remaining blocks** $(i,j)$ — for all $i \ne k$ and $j \ne k$, update:

$$
D_{i,j} \leftarrow \min\!\bigl(D_{i,j},\ D_{i,k} + D_{k,j}\bigr)
$$

Here $D_{i,k}$ and $D_{k,j}$ come from the updated pivot row/column tiles.

This blocking reduces cache misses by reusing submatrices multiple times before moving on, improving performance compared to the naïve Floyd–Warshall algorithm.

---
