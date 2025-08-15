#!/usr/bin/env python3
import os
import random

INF = (1 << 30) - 1
RNG_SEED = 2025

# -------------------------------
# Utilities
# -------------------------------

def create_testcases_dir(dirname="testcases"):
    """Create the testcases directory if it doesn't exist."""
    os.makedirs(dirname, exist_ok=True)
    return dirname

def write_header(f, n: int, m: int):
    """Write the graph header: number of vertices (n) and number of edges (m)."""
    f.write(f"{n} {m}\n")

def rand_weight(wmin=0, wmax=1000):
    """Generate a random integer weight between wmin and wmax (inclusive)."""
    return random.randint(wmin, wmax)

# -------------------------------
# Small/medium graphs (correctness)
# -------------------------------

def add_edge_unique(s: dict, u: int, v: int, w: int):
    """Add an edge only if it is not a self-loop and not a duplicate."""
    if u == v:
        return
    if (u, v) in s:
        return
    s[(u, v)] = w

def line_graph(n, wmin=1, wmax=9):
    """Generate a simple line graph (u -> u+1)."""
    s = {}
    for i in range(n-1):
        add_edge_unique(s, i, i+1, rand_weight(wmin, wmax))
    return list(s.items())

def cycle_graph(n, wmin=1, wmax=9):
    """Generate a simple cycle graph."""
    s = {}
    for i in range(n):
        add_edge_unique(s, i, (i+1) % n, rand_weight(wmin, wmax))
    return list(s.items())

def star_graph(n, center=0, bidir=True, wmin=1, wmax=9):
    """Generate a star-shaped graph with optional bidirectional edges."""
    s = {}
    for i in range(n):
        if i == center:
            continue
        add_edge_unique(s, center, i, rand_weight(wmin, wmax))
        if bidir:
            add_edge_unique(s, i, center, rand_weight(wmin, wmax))
    return list(s.items())

def random_graph(n, m, wmin=0, wmax=1000, allow_bidir=True):
    """Generate a random directed graph with up to m unique edges."""
    s = {}
    tries = 0
    max_tries = max(m * 10, 10000)
    while len(s) < m and tries < max_tries:
        u = random.randrange(n)
        v = random.randrange(n)
        if u != v:
            add_edge_unique(s, u, v, rand_weight(wmin, wmax))
            if allow_bidir and random.random() < 0.2:
                add_edge_unique(s, v, u, rand_weight(wmin, wmax))
        tries += 1
    return list(s.items())

def two_components(n, wmin=1, wmax=9):
    """Generate a graph split into two connected components with no interconnecting edges."""
    s = {}
    A = list(range(0, n//2))
    B = list(range(n//2, n))
    for group in (A, B):
        for i in range(len(group)-1):
            u, v = group[i], group[i+1]
            add_edge_unique(s, u, v, rand_weight(wmin, wmax))
        extra = max(1, len(group)//3)
        for _ in range(extra):
            u = random.choice(group)
            v = random.choice(group)
            if u != v:
                add_edge_unique(s, u, v, rand_weight(wmin, wmax))
    return list(s.items())

def grid_graph_list(rows, cols, wmin=1, wmax=4, bidir=True):
    """Generate a small grid graph and return as an edge list."""
    def vid(r, c):
        return r*cols + c
    n = rows*cols
    s = {}
    for r in range(rows):
        for c in range(cols):
            u = vid(r, c)
            if c+1 < cols:
                v = vid(r, c+1)
                add_edge_unique(s, u, v, rand_weight(wmin, wmax))
                if bidir: add_edge_unique(s, v, u, rand_weight(wmin, wmax))
            if r+1 < rows:
                v = vid(r+1, c)
                add_edge_unique(s, u, v, rand_weight(wmin, wmax))
                if bidir: add_edge_unique(s, v, u, rand_weight(wmin, wmax))
    return n, list(s.items())

# -------------------------------
# Large graphs (performance) — streaming writers
# -------------------------------

def stream_line_graph(path: str, n: int, wmin=1, wmax=9):
    """Write a line graph directly to file without storing in memory."""
    assert 2 <= n <= 40_000
    with open(path, "w") as f:
        write_header(f, n, n-1)
        for u in range(n-1):
            v = u + 1
            w = rand_weight(wmin, wmax)
            f.write(f"{u} {v} {w}\n")

def stream_grid_graph(path: str, rows: int, cols: int, wmin=0, wmax=4, bidir=True):
    """Write a grid graph directly to file without storing in memory."""
    n = rows*cols
    assert 2 <= n <= 40_000
    def vid(r, c): return r*cols + c
    horiz = rows*(cols-1)
    vert  = (rows-1)*cols
    m = horiz + vert
    if bidir:
        m *= 2
    with open(path, "w") as f:
        write_header(f, n, m)
        for r in range(rows):
            for c in range(cols):
                u = vid(r, c)
                if c+1 < cols:
                    v = vid(r, c+1)
                    w = rand_weight(wmin, wmax)
                    f.write(f"{u} {v} {w}\n")
                    if bidir:
                        w2 = rand_weight(wmin, wmax)
                        f.write(f"{v} {u} {w2}\n")
                if r+1 < rows:
                    v = vid(r+1, c)
                    w = rand_weight(wmin, wmax)
                    f.write(f"{u} {v} {w}\n")
                    if bidir:
                        w2 = rand_weight(wmin, wmax)
                        f.write(f"{v} {u} {w2}\n")

def stream_random_dense(path: str, n: int, target_m: int, wmin=0, wmax=9):
    """Write a dense random-like graph directly to file by scanning possible (u,v) pairs."""
    assert 2 <= n <= 40_000
    target_m = min(target_m, n*(n-1))  # respect E ≤ V(V-1)
    with open(path, "w") as f:
        write_header(f, n, target_m)
        written = 0
        u = 0
        v = 1
        while written < target_m:
            if u == v:
                v += 1
                if v == n:
                    u += 1
                    v = 0
                continue
            w = rand_weight(wmin, wmax)
            f.write(f"{u} {v} {w}\n")
            written += 1
            v += 1
            if v == n:
                u += 1
                v = 0
            if u == n:   # wrap just in case
                u = 0

# -------------------------------
# Main
# -------------------------------

def main():
    random.seed(RNG_SEED)
    outdir = create_testcases_dir("testcases")

    # -------- 1–10: correctness (small to medium) --------
    # 1. Minimum valid V (2), no edges
    with open(os.path.join(outdir, "1.in"), "w") as f:
        write_header(f, 2, 0)

    # 2. Two nodes, one directed edge
    with open(os.path.join(outdir, "2.in"), "w") as f:
        write_header(f, 2, 1)
        f.write("0 1 5\n")

    # 3. Line graph, n=4
    n = 4
    edges = line_graph(n, 1, 9)
    with open(os.path.join(outdir, "3.in"), "w") as f:
        write_header(f, n, len(edges))
        for (u, v), w in edges:
            f.write(f"{u} {v} {w}\n")

    # 4. 4-cycle
    n = 4
    edges = cycle_graph(n, 1, 9)
    with open(os.path.join(outdir, "4.in"), "w") as f:
        write_header(f, n, len(edges))
        for (u, v), w in edges:
            f.write(f"{u} {v} {w}\n")

    # 5. Star graph n=8 (with bidirectional edges)
    n = 8
    edges = star_graph(n, center=0, bidir=True, wmin=1, wmax=9)
    with open(os.path.join(outdir, "5.in"), "w") as f:
        write_header(f, n, len(edges))
        for (u, v), w in edges:
            f.write(f"{u} {v} {w}\n")

    # 6. Random sparse graph n=16, m≈2n
    n = 16
    edges = random_graph(n, m=2*n, wmin=0, wmax=50, allow_bidir=True)
    with open(os.path.join(outdir, "6.in"), "w") as f:
        write_header(f, n, len(edges))
        for (u, v), w in edges:
            f.write(f"{u} {v} {w}\n")

    # 7. Random denser graph n=32, m≈6n
    n = 32
    edges = random_graph(n, m=6*n, wmin=0, wmax=50, allow_bidir=True)
    with open(os.path.join(outdir, "7.in"), "w") as f:
        write_header(f, n, len(edges))
        for (u, v), w in edges:
            f.write(f"{u} {v} {w}\n")

    # 8. Two connected components n=64
    n = 64
    edges = two_components(n, 1, 9)
    with open(os.path.join(outdir, "8.in"), "w") as f:
        write_header(f, n, len(edges))
        for (u, v), w in edges:
            f.write(f"{u} {v} {w}\n")

    # 9. 8x8 grid (bidirectional neighbors)
    n_grid, edges = grid_graph_list(8, 8, wmin=1, wmax=4, bidir=True)
    with open(os.path.join(outdir, "9.in"), "w") as f:
        write_header(f, n_grid, len(edges))
        for (u, v), w in edges:
            f.write(f"{u} {v} {w}\n")

    # 10. Sparse larger graph n=128, m≈3n
    n = 128
    edges = random_graph(n, m=3*n, wmin=0, wmax=100, allow_bidir=True)
    with open(os.path.join(outdir, "10.in"), "w") as f:
        write_header(f, n, len(edges))
        for (u, v), w in edges:
            f.write(f"{u} {v} {w}\n")

    # -------- 11–15: performance (each < ~1MB, all constraints respected) --------
    # Keep vertex IDs ≤ 40,000 and weights in [0, 9] to keep files small.

    stream_line_graph(os.path.join(outdir, "11.in"), n=10_000, wmin=0, wmax=9)

    stream_grid_graph(os.path.join(outdir, "12.in"), rows=80, cols=80, wmin=0, wmax=4, bidir=False)

    stream_grid_graph(os.path.join(outdir, "13.in"), rows=60, cols=70, wmin=0, wmax=4, bidir=True)

    # 14. Dense-ish random-like (V = 4,000; E = 70,000) — ~0.85–0.95 MB
    stream_random_dense(os.path.join(outdir, "14.in"), n=4_000, target_m=70_000, wmin=0, wmax=9)

    # 15. Dense-ish random-like (V = 4,500; E = 60,000) — ~0.75–0.9 MB
    stream_random_dense(os.path.join(outdir, "15.in"), n=4_500, target_m=60_000, wmin=0, wmax=9)

    # Print file sizes
    for i in range(1, 16):
        path = os.path.join(outdir, f"{i}.in")
        if os.path.exists(path):
            print(f"{path}: {os.path.getsize(path):,} bytes")

if __name__ == "__main__":
    main()
