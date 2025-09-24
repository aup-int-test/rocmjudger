#!/usr/bin/env python3
import os
import random

def create_testcases_dir():
    if not os.path.exists("testcases"):
        os.makedirs("testcases")

def write_pair_file(filename: str, N: int, preds, targets):
    """寫入單一測資檔案"""
    with open(filename, "w") as f:
        f.write(f"{N}\n")
        f.write(" ".join(map(str, preds)) + "\n")
        f.write(" ".join(map(str, targets)) + "\n")

def gen_random_case(filename: str, N: int):
    """隨機生成 predictions 與 targets"""
    preds   = [round(random.uniform(-1000.0, 1000.0), 2) for _ in range(N)]
    targets = [round(random.uniform(-1000.0, 1000.0), 2) for _ in range(N)]
    write_pair_file(filename, N, preds, targets)

def gen_edge_zero(filename: str, N: int):
    """Edge case: 全部為 0"""
    preds   = [0.0] * N
    targets = [0.0] * N
    write_pair_file(filename, N, preds, targets)

def gen_edge_extreme(filename: str, N: int):
    """Edge case: predictions=1000, targets=-1000"""
    preds   = [1000.0] * N
    targets = [-1000.0] * N
    write_pair_file(filename, N, preds, targets)

def main():
    random.seed(77777)
    create_testcases_dir()

    # 一般隨機測資
    gen_random_case("testcases/1.in", 10)
    gen_random_case("testcases/2.in", 1_000)
    gen_random_case("testcases/3.in", 1_000_000)

    # Edge cases
    gen_edge_zero("testcases/4.in", 100)          # 全 0
    gen_edge_extreme("testcases/5.in", 100)    # 全 1000 / -1000

    # 如果需要超大檔案，可解除下列註解
    gen_random_case("testcases/6.in", 100_000_000)

if __name__ == "__main__":
    main()
