import os
import random

# === Constraints ===
N_MIN, N_MAX = 1, 8
C_MIN, C_MAX = 1, 8
H_MIN, H_MAX = 2, 32
W_MIN, W_MAX = 2, 32
VAL_MIN, VAL_MAX = -100.0, 100.0
# ====================

os.makedirs("testcases", exist_ok=True)

def generate_case(path, N, C, H, W, kernel, stride, pad, seed):
    random.seed(seed)
    with open(path, "w") as f:
        f.write(f"{N} {C} {H} {W} {kernel} {stride} {pad}\n")
        for _ in range(N * C * H * W):
            f.write(f"{random.uniform(VAL_MIN, VAL_MAX):.6f}\n")
    print(f"✅ Generated {path} (N={N}, C={C}, H={H}, W={W}, k={kernel}, s={stride}, p={pad})")

# ---- Normal random cases ----
for i, seed in enumerate([42, 123], start=1):
    N = random.randint(1, 4)
    C = random.randint(1, 4)
    H = random.randint(4, 16)
    W = random.randint(4, 16)
    kernel = random.choice([2, 3])
    stride = random.choice([1, 2])
    pad = random.choice([0, 1])
    generate_case(f"testcases/{i}.in", N, C, H, W, kernel, stride, pad, seed)

# ---- Edge case 1: minimal ----
generate_case("testcases/3.in", 1, 1, 2, 2, 2, 1, 0, 9999)

# ---- Edge case 2: large ----
generate_case("testcases/4.in", 2, 2, 32, 32, 3, 2, 1, 2025)

print("🎯 Generated 4 MaxPooling testcases in ./testcases/")
