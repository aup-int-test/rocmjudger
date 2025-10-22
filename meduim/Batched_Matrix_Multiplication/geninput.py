import os
import random

# create testcases directory
os.makedirs("testcases", exist_ok=True)

def generate_case(path, B, M, N, K, seed):
    random.seed(seed)
    with open(path, "w") as f:
        f.write(f"{B} {M} {N} {K}\n")
        for _ in range(B * M * K):
            f.write(f"{random.uniform(-1000.0, 1000.0):.6f}\n")
        for _ in range(B * K * N):
            f.write(f"{random.uniform(-1000.0, 1000.0):.6f}\n")

# ---- Normal random cases ----
for i, seed in enumerate([42, 123], start=1):
    B = random.randint(1, 128)
    M = random.randint(1, 1024)
    N = random.randint(1, 1024)
    K = random.randint(1, 1024)
    generate_case(f"testcases/{i}.in", B, M, N, K, seed)

# ---- Edge case 1: minimal dimensions ----
# smallest valid problem
generate_case("testcases/3.in", 1, 1, 1, 1, 9999)

# ---- Edge case 2: maximal dimensions ----
# largest valid problem (may be used for stress testing)
generate_case("testcases/4.in", 128, 1024, 1024, 1024, 2025)

print("✅ Generated 4 Batched Matrix Multiplication testcases in ./testcases/")
